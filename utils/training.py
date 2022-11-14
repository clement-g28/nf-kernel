import argparse
import math
import torch

import ffjord_lib.layers.odefunc as odefunc
from utils.dataset import DATASETS, REGRESSION_DATASETS


def ffjord_arguments():
    parser = argparse.ArgumentParser(description="FFJORD Arguments")
    # -------- FFJORD parameters ---------
    SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
    parser.add_argument(
        "--layer_type", type=str, default="concatsquash",
        choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
    )
    parser.add_argument('--dims', type=str, default='64-64-64')
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True)
    parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
    parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

    parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    parser.add_argument('--test_atol', type=float, default=None)
    parser.add_argument('--test_rtol', type=float, default=None)

    parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
    parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)

    # Track quantities
    parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
    parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
    parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
    parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
    parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
    parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")
    # ------------------------------------
    return parser


def training_arguments():
    parser = argparse.ArgumentParser(description="CGlow trainer")
    MODELS = ['cglow', 'seqflow', 'ffjord']
    parser.add_argument("--dataset", type=str, default='mnist', choices=DATASETS + REGRESSION_DATASETS,
                        help="Dataset to use")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--n_epoch", default=1000, type=int, help="number of epoch")
    parser.add_argument("--model", type=str, default='cglow', choices=MODELS, help="Model to use")

    parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
    parser.add_argument("--n_block", default=2, type=int, help="number of blocks")

    # CGLOW parameters
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
    )
    parser.add_argument("--affine", action="store_true", help="use affine coupling instead of additive")

    parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--temp", default=0.7, type=float, help="temperature of sampling")
    parser.add_argument("--n_sample", default=20, type=int, help="number of samples")

    parser.add_argument("--use_tb", action="store_true", help="use tensorboard to save losses")
    parser.add_argument("--save_each_epoch", default=20, type=int, help='save every n epoch')

    parser.add_argument("--fix_mean", action="store_true", help="don\'t learn means")
    parser.add_argument("--with_noise", type=float, default=None, help="add noise to input as learning")
    parser.add_argument("--beta", default=1, type=float, help="distance loss weight")

    # Dataset arguments
    parser.add_argument("--validation", default=0.02, type=float, help="validation rate")

    parser.add_argument("--unique_label", default=None, type=int, help="set if reducing dataset to one label only")
    parser.add_argument("--reduce_class_size", default=None, type=int,
                        help="set to reduce the size of each class in the dataset")
    parser.add_argument("--multi_label", default=None, type=str, help="set if reducing dataset to multi label")

    # Eigenvalues Parameters
    parser.add_argument("--isotrope_gaussian", action="store_true", help="use univariate gaussians")
    parser.add_argument("--uniform_eigval", action="store_true",
                        help='value of uniform eigenvalues associated to the dim-per-label eigenvectors')
    parser.add_argument("--gaussian_eigval", default=None, type=str,
                        help='parameters of the gaussian distribution from which we sample eigenvalues')
    parser.add_argument("--mean_of_eigval", default=10, type=int, help='mean value of eigenvalues')
    parser.add_argument("--set_eigval_manually", default=None, type=str,
                        help='set the eigenvalues manually, it should be in the form of list of n_dim value e.g [50,0.003]')

    parser.add_argument("--dim_per_label", default=None, type=int, help='number of dimension to use for one class')
    return parser


def calc_loss(log_p, logdet, image_size, n_bins):
    n_pixel = image_size * image_size * 3

    loss = -math.log(n_bins) * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (math.log(2) * n_pixel)).mean(),
        (log_p / (math.log(2) * n_pixel)).mean(),
        (logdet / (math.log(2) * n_pixel)).mean(),
    )


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
