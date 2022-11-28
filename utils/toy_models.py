import numpy as np
import math
import torch
import os
import torch.nn as nn

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.density import construct_covariance, multivariate_gaussian
from utils.visualize_flow import visualize_transform
from utils.utils import create_folder, calculate_log_p_with_gaussian_params, \
    calculate_log_p_with_gaussian_params_regression
from utils.ffjord_model import build_model_tabular, create_regularization_fns, add_spectral_norm, set_cnf_options, \
    get_regularization, spectral_norm_power_iteration
from utils.seqflow_model import load_flow_model


# abstract NF class
class NF(nn.Module):
    def __init__(self, model, gaussian_params=None, device='cuda:0', learn_mean=True, reg_use_var=False, dataset=None):
        super().__init__()
        if dataset is not None and dataset.is_regression_dataset():
            self.dataset_type = 'regression'
            self.label_min = np.min(dataset.true_labels)
            self.label_max = np.max(dataset.true_labels)

            self.use_var = reg_use_var
            if reg_use_var:
                assert len(np.where(gaussian_params[0][2] != 1)[
                               0]) == 0, 'eigenval should be 1 everywhere to use variance in loss '
                self.label_mindist = dataset.label_mindist

                def calculate_log_p_reg(x, label, means, gaussian_params):
                    mean, inv_cov, det = self.get_regression_gaussian_training_parameters_var(label, means)
                    return calculate_log_p_with_gaussian_params_regression(x, mean, inv_cov, det)
            else:
                def calculate_log_p_reg(x, label, means, gaussian_params):
                    mean, inv_cov, det = self.get_regression_gaussian_training_parameters(label, means)
                    return calculate_log_p_with_gaussian_params_regression(x, mean, inv_cov, det)

            self.calculate_log_p = calculate_log_p_reg
            self.sample_from_distrib = self.sample_from_distrib_regression
            self.vis_log_p_calcul = self.vis_log_p_calcul_regression
        else:
            self.dataset_type = 'labeled'
            self.calculate_log_p = calculate_log_p_with_gaussian_params
            self.sample_from_distrib = self.sample_from_distrib_label
            self.vis_log_p_calcul = self.vis_log_p_calcul_label
        self.device = device
        self.model = model
        self.gp = gaussian_params
        if gaussian_params is not None:
            # self.means = []
            means = []
            self.gaussian_params = []
            self.eigvals = []
            self.eigvecs = []
            for gaussian_param in gaussian_params:
                mean = gaussian_param[0]
                eigenvec = gaussian_param[1]
                eigenval = gaussian_param[2]
                self.dim_per_lab = np.count_nonzero(eigenval > 1)
                covariance_matrix = construct_covariance(eigenvec, eigenval)
                determinant = np.linalg.det(covariance_matrix)
                inverse_matrix = np.linalg.inv(covariance_matrix)
                means.append(torch.from_numpy(mean).unsqueeze(0))
                self.gaussian_params.append((torch.from_numpy(inverse_matrix).to(device),
                                             determinant))
                indexes = np.argsort(-eigenval, kind='mergesort')
                eigenvec = eigenvec[indexes]
                eigenval = eigenval[indexes]
                self.eigvals.append(torch.from_numpy(eigenval).unsqueeze(0).to(device))
                self.eigvecs.append(torch.from_numpy(eigenvec).unsqueeze(0).to(device))
            self.eigvals = torch.cat(self.eigvals, dim=0).to(torch.float32)
            self.eigvecs = torch.cat(self.eigvecs, dim=0).to(torch.float32)
            if learn_mean:
                self.means = nn.Parameter(torch.cat(means, dim=0))
            else:
                self.means = torch.cat(means, dim=0).to(device)

    def calc_last_z_shape(self, input_size):
        return (input_size,)

    def downstream_process(self):
        raise NotImplementedError

    def upstream_process(self, loss):
        raise NotImplementedError

    def forward(self, input, label, pair_with_noise=False):
        raise NotImplementedError

    def reverse(self, z, reconstruct=False):
        raise NotImplementedError

    def sample_from_distrib_label(self, n, dim):
        n_per_distrib = math.ceil(n / len(self.gaussian_params))
        samples = []
        for i, gaussian_param in enumerate(self.gp):
            mean = self.means[i].detach().cpu().numpy()
            eigenvec = gaussian_param[1]
            eigenval = gaussian_param[2]
            cov = construct_covariance(eigenvec, eigenval)
            z = np.random.multivariate_normal(mean, cov, n_per_distrib)
            z_sample = torch.from_numpy(z).reshape(n_per_distrib, -1).float().to(self.device)
            samples.append(z_sample)
        samples = torch.cat(samples, dim=0)
        return samples

    def sample_from_distrib_regression(self, n, dim):
        t_means = self.means.detach().cpu()
        sampled_fac = torch.rand(n).unsqueeze(1).repeat(1, dim)
        sampled_means = t_means[0] * sampled_fac + t_means[1] * (1 - sampled_fac)
        # samples = torch.normal(mean=sampled_means)
        if self.use_var:
            variance = self.label_mindist / (self.label_max - self.label_min)
            variance = variance * (t_means[1] - t_means[0])
            variance[np.where(variance <= 0)] = 1
        else:
            variance = torch.from_numpy(self.gp[0][2])
        samples = torch.normal(mean=sampled_means) * variance.unsqueeze(0)

        return samples

    def vis_log_p_calcul_label(self, z):
        z = z.to(torch.float64)

        log_ps = []
        for i, gaussian_param in enumerate(self.gaussian_params):
            log_ps.append(multivariate_gaussian(z, mean=self.means[i], determinant=gaussian_param[1],
                                                inverse_covariance_matrix=gaussian_param[0]).unsqueeze(1))
        log_p = torch.max(torch.cat(log_ps, dim=1), dim=1).values

        return log_p

    def vis_log_p_calcul_regression(self, z):
        from utils.testing import project_between
        means = self.means.detach().cpu().numpy()
        np_z = z.detach().cpu().numpy()
        proj, _ = project_between(np_z, means[0], means[1])
        s_means = torch.from_numpy(proj).to(z.device)

        # z = z.to(torch.float64)
        # dir = (self.means[-1] - self.means[0]).to(torch.float64)
        # dot_dir = torch.dot(dir, dir)
        # lab_fac = torch.mm(z, dir.unsqueeze(1)) / dot_dir
        # s_means = lab_fac * dir

        # mins = torch.min(self.means, axis=0)[0]
        # maxs = torch.max(self.means, axis=0)[0]
        # li = []
        # for dim in range(mins.shape[0]):
        #     li.append(torch.clamp(s_means[:, dim], mins[dim], maxs[dim]).unsqueeze(1))
        # s_means = torch.cat(li, axis=1)

        if self.use_var:
            variance = self.label_mindist / (self.label_max - self.label_min)
            variance = variance * (self.means[1] - self.means[0])
            variance = torch.where(variance <= 0, torch.ones_like(variance), variance)
            inv_cov = self.gaussian_params[0][0] * (1 / variance)
            det = self.gaussian_params[0][1]
            for v in variance:
                det = det * v
        else:
            det = self.gaussian_params[0][1]
            inv_cov = self.gaussian_params[0][0]

        log_p = multivariate_gaussian(z, mean=s_means, determinant=det,
                                      inverse_covariance_matrix=inv_cov).unsqueeze(1)

        return log_p

    @staticmethod
    def get_transforms(model):
        raise NotImplementedError

    def sample_evaluation(self, itr, train_dataset, val_dataset, z_shape, save_dir, writer=None,
                          batch_size=200):
        raise NotImplementedError

    def evaluate(self, itr, x, y, z, dataset, save_dir, writer=None):
        return -1
        # fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        # x = x.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()
        # z = z.detach().cpu().numpy()
        # np_means = self.means.detach().cpu().numpy()
        # z = np.concatenate((z, np_means), axis=0)
        # ax1.scatter(x[:, 0], x[:, 1], c=y)
        # add_col = [int(np.max(dataset.true_labels)) + i + 1 for i in range(np_means.shape[0])]
        # y = np.concatenate((y, np.array(add_col)), axis=0)
        # ax2.scatter(z[:, 0], z[:, 1], c=y)
        #
        # fig_filename = os.path.join(save_dir, 'figs', 'z_{:04d}.jpg'.format(itr))
        # create_folder(os.path.dirname(fig_filename))
        # means_title = str(np.around(np_means, 2).tolist())
        # means_dist = str(torch.cdist(self.means, self.means).mean().detach().cpu().numpy().item())
        # plt.title(means_title + ', d=' + means_dist)
        # plt.savefig(fig_filename)
        # plt.close()

    def get_regression_gaussian_mean(self, label, means):
        lab_fac = ((label - self.label_min) / (self.label_max - self.label_min)).unsqueeze(1)
        mean = means[0] * lab_fac + means[1] * (1 - lab_fac)
        return mean

    def get_regression_gaussian_training_parameters(self, label, means):
        inv_cov = self.gaussian_params[0][0]
        det = self.gaussian_params[0][1]

        mean = self.get_regression_gaussian_mean(label, means)
        return mean, inv_cov, det

    def get_regression_gaussian_training_parameters_var(self, label, means):
        # variance = self.label_mindist / (self.label_max - self.label_min)
        # variance = variance * (means[1] - means[0]).abs() *100
        # variance[np.where(variance <= 0)] = 1
        # variance = torch.where(variance <= 0, torch.ones_like(variance)*0.1, variance)
        variance = torch.ones_like(means[0])*0.1
        if (means[1] - means[0]).abs().mean() > 6:
            variance = self.label_mindist / (self.label_max - self.label_min)
            variance = variance * (means[1] - means[0]).abs()
        # variance = torch.ones_like(variance)*0.1
        inv_cov = self.gaussian_params[0][0] * (1 / variance)
        det = self.gaussian_params[0][1]
        for v in variance:
            det = det * v

        mean = self.get_regression_gaussian_mean(label, means)
        return mean, inv_cov, det

    def get_regression_gaussian_sampling_parameters(self, label):
        eigenvec = self.gp[0][1]
        eigenval = self.gp[0][2]
        covariance_matrix = construct_covariance(eigenvec, eigenval)
        means = self.means.detach().cpu().numpy()
        if self.use_var:
            variance = self.label_mindist / (self.label_max - self.label_min)
            variance = variance * (means[1] - means[0]) + .001
            covariance_matrix = covariance_matrix * variance

        mean = self.get_regression_gaussian_mean(label)
        return mean, covariance_matrix


def load_seqflow_model(num_inputs, n_flows, gaussian_params, learn_mean=True, device='cuda:0', reg_use_var=False,
                       dataset=None):
    # Flow Sequential
    num_hidden = num_inputs * 4
    num_cond_inputs = None
    model = load_flow_model(num_inputs, n_flows, num_hidden, num_cond_inputs, device)

    model.to(torch.float).to(device)
    model.train()
    model = SeqFlow(model, gaussian_params=gaussian_params, learn_mean=learn_mean, reg_use_var=reg_use_var,
                    dataset=dataset)
    return model


class SeqFlow(NF):
    def __init__(self, model, gaussian_params=None, device='cuda:0', learn_mean=True, reg_use_var=False, dataset=None):
        super().__init__(model, gaussian_params, device, learn_mean, reg_use_var, dataset)

    def downstream_process(self):
        return -1

    def upstream_process(self, loss):
        return loss

    def forward(self, input, label, pair_with_noise=False):
        z, delta_logp = self.model(input)

        # compute distance loss
        distloss = torch.log(1 + torch.cdist(self.means, self.means).mean())
        # compute log q(z)
        # means = torch.empty_like(self.means).copy_(self.means)
        log_p = self.calculate_log_p(z, label, self.means, self.gaussian_params)

        return log_p, distloss, delta_logp, z

    def reverse(self, z, reconstruct=False):
        inputs, logdets = self.model(z, cond_inputs=None, mode='inverse', logdets=None)

        return inputs

    @staticmethod
    def get_transforms(model):
        def sample_fn(z, logpz=None):
            res, logp = model(z, mode='inverse')
            if logpz is not None:
                return res, logp
            else:
                return res

        def density_fn(x, logpx=None):
            res, logp = model(x)
            if logpx is not None:
                return res, logp
            else:
                return res

        return sample_fn, density_fn

    def sample_evaluation(self, itr, train_dataset, val_dataset, z_shape, save_dir, writer=None,
                          batch_size=200):
        idx = np.random.randint(0, val_dataset.X.shape[0], batch_size)
        p_samples, y = val_dataset[idx]
        p_samples = p_samples.numpy()

        sample_fn, density_fn = self.get_transforms(self.model)

        plt.figure(figsize=(9, 3))
        visualize_transform(
            p_samples, self.sample_from_distrib, self.vis_log_p_calcul, transform=sample_fn,
            inverse_transform=density_fn, samples=True, npts=100, device=self.device
        )
        fig_filename = os.path.join(save_dir, 'figs', '{:04d}.jpg'.format(itr))
        create_folder(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()

        p_samples = torch.from_numpy(p_samples).float().to(self.device)
        z, _ = self.model(p_samples)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        x = p_samples.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        np_means = self.means.detach().cpu().numpy()
        z = np.concatenate((z, np_means), axis=0)
        lab_min = np.min(val_dataset.true_labels)
        lab_max = np.max(val_dataset.true_labels)
        ax1.scatter(x[:, 0], x[:, 1], c=y, vmin=lab_min, vmax=lab_max)
        # add_col = [int(np.max(val_dataset.true_labels)) + i + 1 for i in range(np_means.shape[0])]
        add_col = [lab_min, lab_max]
        y = np.concatenate((y, np.array(add_col)), axis=0)
        ax2.scatter(z[:, 0], z[:, 1], c=y, vmin=lab_min, vmax=lab_max)

        fig_filename = os.path.join(save_dir, 'figs', 'z_{:04d}.jpg'.format(itr))
        create_folder(os.path.dirname(fig_filename))
        means_title = str(np.around(np_means, 2).tolist())
        means_dist = str(torch.cdist(self.means, self.means).mean().detach().cpu().numpy().item())
        plt.title(means_title + ', d=' + means_dist)
        plt.savefig(fig_filename)
        plt.close()


def load_ffjord_model(args, n_dim, gaussian_params, learn_mean=True, device='cuda:0', reg_use_var=False, dataset=None):
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, n_dim, regularization_fns).to(device)
    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(args, model)

    model.to(torch.float).to(device)
    model.train()
    model = FFJORD(args, model, regularization_coeffs=regularization_coeffs, gaussian_params=gaussian_params,
                   learn_mean=learn_mean, reg_use_var=reg_use_var, dataset=dataset)
    return model


class FFJORD(NF):
    def __init__(self, args, model, regularization_coeffs, gaussian_params=None, device='cuda:0', learn_mean=True,
                 reg_use_var=False, dataset=None):
        super().__init__(model, gaussian_params, device, learn_mean, reg_use_var, dataset)
        self.regularization_coeffs = regularization_coeffs
        self.spectral_norm = args.spectral_norm

    def downstream_process(self):
        if self.spectral_norm: spectral_norm_power_iteration(self.model, 1)

    def upstream_process(self, loss):
        if len(self.regularization_coeffs) > 0:
            reg_states = get_regularization(self.model, self.regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, self.regularization_coeffs) if coeff != 0
            )
            loss = loss + reg_loss
        return loss

    def forward(self, input, label, pair_with_noise=False):
        zero = torch.zeros(input.shape[0], 1).to(input)
        z, delta_logp = self.model(input, zero)

        # compute log q(z)
        log_p = self.calculate_log_p(z, label, self.means, self.gaussian_params)
        distloss = torch.log(1 + torch.cdist(self.means, self.means).mean())

        return log_p, distloss, -delta_logp, z

    def reverse(self, z, reconstruct=False):
        return self.model(z, reverse=True)

    @staticmethod
    def get_transforms(model):
        def sample_fn(z, logpz=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z, reverse=True)

        def density_fn(x, logpx=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

        return sample_fn, density_fn

    def sample_evaluation(self, itr, train_dataset, val_dataset, z_shape, save_dir, writer=None,
                          batch_size=200):
        idx = np.random.randint(0, val_dataset.X.shape[0], batch_size)
        p_samples, y = val_dataset[idx]
        p_samples = p_samples.numpy()

        sample_fn, density_fn = self.get_transforms(self.model)

        plt.figure(figsize=(9, 3))
        visualize_transform(
            p_samples, self.sample_from_distrib, self.vis_log_p_calcul, transform=sample_fn,
            inverse_transform=density_fn, samples=True, npts=100, device=self.device
        )
        fig_filename = os.path.join(save_dir, 'figs', '{:04d}.jpg'.format(itr))
        create_folder(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()

        p_samples = torch.from_numpy(p_samples).float().to(self.device)
        z = self.model(p_samples)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        x = p_samples.detach().cpu().numpy()
        z = z.detach().cpu().numpy()
        np_means = self.means.detach().cpu().numpy()
        z = np.concatenate((z, np_means), axis=0)
        ax1.scatter(x[:, 0], x[:, 1], c=y)
        add_col = [int(np.max(val_dataset.true_labels)) + i + 1 for i in range(np_means.shape[0])]
        y = np.concatenate((y, np.array(add_col)), axis=0)
        ax2.scatter(z[:, 0], z[:, 1], c=y)

        fig_filename = os.path.join(save_dir, 'figs', 'z_{:04d}.jpg'.format(itr))
        create_folder(os.path.dirname(fig_filename))
        means_title = str(np.around(np_means, 2).tolist())
        means_dist = str(torch.cdist(self.means, self.means).mean().detach().cpu().numpy().item())
        plt.title(means_title + ', d=' + means_dist)
        plt.savefig(fig_filename)
        plt.close()
