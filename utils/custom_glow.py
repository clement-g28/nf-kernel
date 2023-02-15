import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy import linalg as la

from utils.density import construct_covariance
from utils.utils import calculate_log_p_with_gaussian_params

from torchvision import utils

logabs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                    .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet

        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        logdet = (
                height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape

        weight = self.calc_weight()

        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
                self.w_p
                @ (self.w_l * self.l_mask + self.l_eye)
                @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()

        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()

        self.affine = affine

        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = F.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()

        self.actnorm = ActNorm(in_channel)

        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)

        else:
            self.invconv = InvConv2d(in_channel)

        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)

        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2

        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)

        return input


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))

        self.split = split

        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)

        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0

        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        return out, logdet

    def reverse(self, output, eps=None, reconstruct=False):
        input = output

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape

        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )

        return unsqueezed


# class CGlow(nn.Module):
#     def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True, gaussian_params=None, device='cuda:0',
#                  learn_mean=True):
#         super().__init__()
#         self.gp = gaussian_params
#         self.device = device
#         if gaussian_params is not None:
#             # self.means = []
#             means = []
#             self.gaussian_params = []
#             self.eigvals = []
#             self.eigvecs = []
#             for gaussian_param in gaussian_params:
#                 mean = gaussian_param[0]
#                 eigenvec = gaussian_param[1]
#                 eigenval = gaussian_param[2]
#                 self.dim_per_lab = np.count_nonzero(eigenval > 1)
#                 covariance_matrix = construct_covariance(eigenvec, eigenval)
#                 determinant = np.linalg.det(covariance_matrix)
#                 inverse_matrix = np.linalg.inv(covariance_matrix)
#                 means.append(torch.from_numpy(mean).unsqueeze(0))
#                 self.gaussian_params.append((torch.from_numpy(inverse_matrix).to(device),
#                                              determinant))
#                 indexes = np.argsort(-eigenval, kind='mergesort')
#                 eigenvec = eigenvec[indexes]
#                 eigenval = eigenval[indexes]
#                 self.eigvals.append(torch.from_numpy(eigenval).unsqueeze(0).to(device))
#                 self.eigvecs.append(torch.from_numpy(eigenvec).unsqueeze(0).to(device))
#             self.eigvals = torch.cat(self.eigvals, dim=0).to(torch.float32)
#             self.eigvecs = torch.cat(self.eigvecs, dim=0).to(torch.float32)
#             if learn_mean:
#                 self.means = nn.Parameter(torch.cat(means, dim=0))
#             else:
#                 self.means = torch.cat(means, dim=0).to(device)
#
#         self.blocks = nn.ModuleList()
#         self.n_channel = in_channel
#         self.n_block = n_block
#         for i in range(n_block - 1):
#             self.blocks.append(
#                 Block(in_channel, n_flow, affine=affine, conv_lu=conv_lu))
#             in_channel *= 4
#         self.blocks.append(Block(in_channel, n_flow, split=False, affine=affine))
#
#     def calc_last_z_shape(self, input_size):
#         n_channel = self.n_channel
#         for i in range(self.n_block - 1):
#             input_size //= 2
#             n_channel *= 4
#
#         input_size //= 2
#         z_shape = (n_channel * 4, input_size, input_size)
#
#         return z_shape
#
#     def forward(self, input, label, pair_with_noise=False):
#         logdet = 0
#         out = input
#
#         for block in self.blocks:
#             out, det = block(out)
#             logdet = logdet + det
#
#         log_p = calculate_log_p_with_gaussian_params(out, label, self.means, self.gaussian_params)
#         distloss = torch.log(1 + torch.cdist(self.means, self.means).mean())
#
#         if pair_with_noise:
#             # Div by 2 (z, noisy_z)
#             d = int(out.shape[0] / 2)
#             ori_z = out[:d]
#             noisy_z = out[d:]
#             ori_lab = label[:d]
#             noisy_lab = label[d:]
#             loss = denoise_loss(ori_z, noisy_z, ori_lab, noisy_lab, self.means, self.eigvals, self.eigvecs,
#                                 self.dim_per_lab)
#             return log_p, distloss, loss, logdet, out
#
#         return log_p, distloss, logdet, out
#
#     def reverse(self, z, reconstruct=False):
#         for i, block in enumerate(self.blocks[::-1]):
#             if i == 0:
#                 input = block.reverse(z, z, reconstruct=reconstruct)
#
#             else:
#                 input = block.reverse(input, z, reconstruct=reconstruct)
#
#         return input
#
#     def sample_evaluation(self, itr, train_dataset, val_dataset, z_shape, save_dir, writer=None):
#         z_samples = []
#         for i, gaussian_param in enumerate(self.gp):
#             mean = self.means[i].detach().cpu().numpy().squeeze()
#             eigenvec = gaussian_param[1]
#             eigenval = gaussian_param[2]
#             cov = construct_covariance(eigenvec, eigenval)
#             z_sample = np.expand_dims(np.random.multivariate_normal(mean, cov).reshape(z_shape), axis=0)
#             z_samples.append(z_sample)
#         z_sample = torch.from_numpy(np.concatenate(z_samples, axis=0)).float().to(self.device)
#
#         images = self.reverse(z_sample).cpu().data
#         images = train_dataset.rescale(images)
#         utils.save_image(
#             images,
#             f"{save_dir}/sample/{str(itr + 1).zfill(6)}.png",
#             normalize=True,
#             nrow=10,
#             range=(0, 255),
#         )
#         if writer is not None:
#             img_grid = utils.make_grid(images, nrow=10, padding=2, pad_value=0, normalize=True,
#                                        range=(0, 255), scale_each=False)
#             writer.add_image('sample', img_grid)
#
#     def evaluate(self, itr, x, y, z, dataset, save_dir, writer=None):
#         images = self.reverse(z).cpu().data
#         images = dataset.rescale(images)
#         utils.save_image(
#             images,
#             f"{save_dir}/sample/val_{str(itr + 1).zfill(6)}.png",
#             normalize=True,
#             nrow=10,
#             range=(0, 255),
#         )
#         if writer is not None:
#             img_grid = utils.make_grid(images, nrow=10, padding=2, pad_value=0, normalize=True,
#                                        range=(0, 255), scale_each=False)
#             writer.add_image('val', img_grid)

class CGlow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.n_channel = in_channel
        self.n_block = n_block
        for i in range(n_block - 1):
            self.blocks.append(
                Block(in_channel, n_flow, affine=affine, conv_lu=conv_lu))
            in_channel *= 4
        self.blocks.append(Block(in_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        logdet = 0
        out = input

        for block in self.blocks:
            out, det = block(out)
            logdet = logdet + det

        return out, logdet

    def reverse(self, z, reconstruct=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z, z, reconstruct=reconstruct)

            else:
                input = block.reverse(input, z, reconstruct=reconstruct)

        return input


class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x):
        return self.module(x)

