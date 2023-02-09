import random
import torch
import os
import math
import numpy as np
from PIL import Image

from utils.density import multivariate_gaussian

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_dict_to_tensorboard(writer, val_dict, base_name, iteration):
    for name, val in val_dict.items():
        if isinstance(val, dict):
            write_dict_to_tensorboard(writer, val, base_name=base_name + "/" + name, iteration=iteration)
        elif isinstance(val, (list, np.ndarray)):
            continue
        elif isinstance(val, (int, float)):
            writer.add_scalar(base_name + "/" + name, val, iteration)
        else:
            print("Skipping output \"" + str(name) + "\" of value " + str(val) + "(%s)" % (val.__class__.__name__))


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def save_every_pic(path, numpy_pics, methods, labels, add_str=None, clamp_min=0, clamp_max=255):
    create_folder(path)
    add_str = '' if add_str is None else f'_{add_str}'
    numpy_pics = np.clip(numpy_pics, clamp_min, clamp_max)
    for i, pic in enumerate(numpy_pics):
        im = pic.squeeze().astype(np.uint8)
        im = Image.fromarray(im, mode='L')
        name = f"{methods[i]}_{str(labels[i])}{add_str}.png"
        im.save(f'{path}/{name}')


def save_fig(X, labels, save_path, limits=None, size=7, rangelabels=None, eps=False):
    if rangelabels is not None:
        vmin = rangelabels[0]
        vmax = rangelabels[-1]
        plt.scatter(X[:, 0], X[:, 1], s=size, c=labels, zorder=3, vmin=vmin, vmax=vmax)
    else:
        plt.scatter(X[:, 0], X[:, 1], s=size, c=labels, zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")

    if limits:
        x_xmin, x_xmax, x_ymin, x_ymax = limits
        plt.xlim([x_xmin - 1, x_xmax + 1])
        plt.ylim([x_ymin - 1, x_ymax + 1])

    if eps:
        plt.savefig(fname=f'{save_path}.eps', format='eps')
    plt.savefig(fname=f'{save_path}.png', format='png')

    plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_gaussian_params(dataset, al_list, isotrope=False, dim_per_label=30, fixed_eigval=None):
    uni = np.unique(dataset.true_labels)
    n_dim = dataset.get_n_dim()
    gaussian_params = []
    if isotrope:
        for i, label in enumerate(uni):
            mean = np.zeros(n_dim)
            eigenvecs = np.zeros((n_dim, n_dim))
            np.fill_diagonal(eigenvecs, 1)
            eigenvals = np.ones(n_dim)
            gaussian_params.append((mean, eigenvecs, eigenvals, label))
    else:
        for i, label in enumerate(uni):
            mean = np.zeros(n_dim)
            eigenvecs = np.zeros((n_dim, n_dim))
            np.fill_diagonal(eigenvecs, 1)
            if fixed_eigval is None:
                be = np.exp(
                    1 / (n_dim - dim_per_label) * np.log(1 / math.pow(sum(al_list) / len(al_list), dim_per_label)))
                eigenvals = np.ones(n_dim) * be
                eigenvals[dim_per_label * i:dim_per_label * (i + 1)] = al_list
            else:
                eigenvals = np.ones(n_dim)
                eigenvals[:] = fixed_eigval
            gaussian_params.append((mean, eigenvecs, eigenvals, label))

    return gaussian_params


def initialize_regression_gaussian_params(dataset, al_list, isotrope=False, dim_per_label=30, fixed_eigval=None):
    n_dim = dataset.get_n_dim()
    gaussian_params = []
    if isotrope:
        for i in range(2):
            mean = np.zeros(n_dim)
            # mean = np.ones(n_dim) * -20 * i
            eigenvecs = np.zeros((n_dim, n_dim))
            np.fill_diagonal(eigenvecs, 1)
            eigenvals = np.ones(n_dim)
            eigenvals[:] = al_list
            gaussian_params.append((mean, eigenvecs, eigenvals, i))
    else:
        assert n_dim != dim_per_label, 'for regression dataset, isotrope_gaussian should be used if n_dim_per_label ' \
                                       'is not defined and therefore n_dim_per_label = n_dim'
        for i in range(2):
            mean = np.zeros(n_dim)
            eigenvecs = np.zeros((n_dim, n_dim))
            np.fill_diagonal(eigenvecs, 1)
            if fixed_eigval is None:
                be = np.exp(
                    1 / (n_dim - dim_per_label) * np.log(1 / math.pow(sum(al_list) / len(al_list), dim_per_label)))
                eigenvals = np.ones(n_dim) * be
                eigenvals[dim_per_label * i:dim_per_label * (i + 1)] = al_list
            else:
                eigenvals = np.ones(n_dim)
                eigenvals[:] = fixed_eigval
            gaussian_params.append((mean, eigenvecs, eigenvals, i))

    return gaussian_params


def initialize_tmp_regression_gaussian_params(dataset, al_list, ones=False):
    n_dim = dataset.X[0].shape[0]
    for sh in dataset.X[0].shape[1:]:
        n_dim *= sh
    gaussian_params = []

    assert len(al_list) == 1, 'al_list should contains one dimension only'
    dim_interpolation = 1

    for i in range(2):
        mean = np.zeros(n_dim)
        eigenvecs = np.zeros((n_dim, n_dim))
        np.fill_diagonal(eigenvecs, 1)
        be = np.exp(
            1 / (n_dim - dim_interpolation) * np.log(1 / math.pow(sum(al_list) / len(al_list), dim_interpolation)))
        eigenvals = np.ones(n_dim) * be if not ones else np.ones(n_dim) * 0.1
        eigenvals[:1] = al_list
        mean[:1] = i
        gaussian_params.append((mean, eigenvecs, eigenvals, i))

    return gaussian_params


def calculate_log_p_with_gaussian_params(x, label, means, gaussian_params):
    log_ps = []
    for i, gaussian_param in enumerate(gaussian_params):
        log_ps.append(multivariate_gaussian(x, mean=means[i], determinant=gaussian_param[1],
                                            inverse_covariance_matrix=gaussian_param[0]).unsqueeze(1))

    log_ps = torch.cat(log_ps, dim=1)
    one_hot_label = torch.nn.functional.one_hot(label, num_classes=log_ps.shape[1])
    log_p = torch.sum(log_ps * one_hot_label, dim=1)

    return log_p


# def calculate_log_p_with_gaussian_params_regression(x, label, means, gaussian_params, label_min, label_max):
#     log_ps = []
#     for i, gaussian_param in enumerate(gaussian_params):
#         log_ps.append(multivariate_gaussian(x, mean=means[i], determinant=gaussian_param[1],
#                                             inverse_covariance_matrix=gaussian_param[0]).unsqueeze(1))
#
#     lab_fac = ((label - label_min) / (label_max - label_min)).unsqueeze(1)
#     log_p = log_ps[0] * lab_fac + log_ps[1] * (1 - lab_fac)
#
#     return log_p


def calculate_log_p_with_gaussian_params_regression(x, mean, inv_cov, det):
    log_ps = multivariate_gaussian(x, mean=mean, determinant=det,
                                   inverse_covariance_matrix=inv_cov).unsqueeze(1)

    return log_ps
