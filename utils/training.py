import argparse
import math
import torch

import ffjord_lib.layers.odefunc as odefunc
from utils.dataset import DATASETS, REGRESSION_DATASETS, GRAPH_DATASETS
from utils.models import GRAPH_MODELS, SIMPLE_MODELS, IMAGE_MODELS


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def graphnvp_arguments():
    parser = argparse.ArgumentParser(description="GraphNVP Arguments")

    # reproducibility
    # parser.add_argument('--num_atoms', type=int, default=9, help='Maximum number of atoms in a molecule')
    # parser.add_argument('--num_rels', type=int, default=4, help='Number of bond types')
    # parser.add_argument('--num_atom_types', type=int, default=4, help='Types of atoms that can be used in a molecule')
    parser.add_argument('--num_node_masks', type=int, default=9,
                        help='Number of node masks to be used in coupling layers')
    parser.add_argument('--num_channel_masks', type=int, default=4,
                        help='Number of channel masks to be used in coupling layers')
    parser.add_argument('--num_node_coupling', type=int, default=12, help='Number of coupling layers with node masking')
    parser.add_argument('--num_channel_coupling', type=int, default=6,
                        help='Number of coupling layers with channel masking')
    parser.add_argument('--node_mask_size', type=int, default=5, help='Number of cells to be masked in the Node '
                                                                      'coupling layer')
    parser.add_argument('--channel_mask_size', type=int, default=-1, help='Number of cells to be masked in the Channel '
                                                                          'coupling layer')
    parser.add_argument('--apply_batch_norm', type=bool, default=False, help='Whether batch '
                                                                             'normalization should be performed')
    # parser.add_argument('--additive_transformations', type=bool, default=True,
    #                     help='if True, apply only addictive coupling layers; else, apply affine coupling layers')
    parser.add_argument('--additive_transformations', action='store_true', default=False,
                        help='if True, apply only addictive coupling layers; else, apply affine coupling layers')
    # Model configuration.
    parser.add_argument('--use_switch', type=bool, default=False, help='use switch activation for R-GCN')
    parser.add_argument('--num_gcn_layer', type=int, default=3)

    # Generation args
    parser.add_argument('--min_atoms', type=int, default=2, help='Minimum number of atoms in a generated molecule')
    parser.add_argument('--num_gen', type=int, default=100,
                        help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--min_gen_epoch', type=int, default=5,
                        help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--max_resample', type=int, default=200, help='the times of resampling each epoch')
    parser.add_argument('--atomic_num_list', type=int, default=[6, 7, 8, 9, 0],
                        help='atomic number list for datasets')
    # parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'],
    #                    help='TODO: postprocessing to convert continuous A and X to discrete ones')

    # Added, hidden channels params
    parser.add_argument('--mlp_channels', type=str, default="128,128",
                        help='Hidden channel list for bonds tensor, delimited list input ')  # 256,256 qm9
    parser.add_argument('--gnn_channels_gcn', type=str, default="8,64",
                        help='Hidden dimension list for graph convolution for atoms matrix, delimited list input ')
    parser.add_argument('--gnn_channels_hidden', type=str, default="64,128",
                        help='Hidden dimension list for linear transformation for atoms, delimited list input ')

    return parser


def moflow_arguments():
    parser = argparse.ArgumentParser(description="MoFlow Arguments")

    # For bonds
    parser.add_argument('--b_n_flow', type=int, default=10,
                        help='Number of masked glow coupling layers per block for bond tensor')
    parser.add_argument('--b_n_block', type=int, default=1, help='Number of glow blocks for bond tensor')
    parser.add_argument('--b_hidden_ch', type=str, default="128,128",
                        help='Hidden channel list for bonds tensor, delimited list input ')
    parser.add_argument('--b_conv_lu', type=int, default=1, choices=[0, 1, 2],
                        help='0: InvConv2d for 1*1 conv, 1:InvConv2dLU for 1*1 conv, 2: No 1*1 conv, '
                             'swap updating in the coupling layer')
    # For atoms
    parser.add_argument('--a_n_flow', type=int, default=27,
                        help='Number of masked flow coupling layers per block for atom matrix')
    parser.add_argument('--a_n_block', type=int, default=1, help='Number of flow blocks for atom matrix')
    parser.add_argument('--a_hidden_gnn', type=str, default="64,",
                        help='Hidden dimension list for graph convolution for atoms matrix, delimited list input ')
    parser.add_argument('--a_hidden_lin', type=str, default="128,64",
                        help='Hidden dimension list for linear transformation for atoms, delimited list input ')
    parser.add_argument('--mask_row_size_list', type=str, default="1,",
                        help='Mask row size list for atom matrix, delimited list input ')
    parser.add_argument('--mask_row_stride_list', type=str, default="1,",
                        help='Mask row stride list for atom matrix, delimited list input')
    # General
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--learn_dist', type=strtobool, default='true', help='learn the distribution of feature matrix')
    parser.add_argument('--noise_scale', type=float, default=0.6, help='x + torch.rand(x.shape) * noise_scale')

    return parser


def ffjord_arguments():
    parser = argparse.ArgumentParser(description="FFJORD Arguments")

    parser.add_argument("--n_block", default=2, type=int, help="number of blocks")

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


def cglow_arguments():
    parser = argparse.ArgumentParser(description="CGlow Arguments")

    parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
    parser.add_argument("--n_block", default=2, type=int, help="number of blocks")
    # CGLOW parameters
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
    )
    parser.add_argument("--affine", action="store_true", help="use affine coupling instead of additive")

    return parser


def seqflow_arguments():
    parser = argparse.ArgumentParser(description="SeqFlow Arguments")
    parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
    return parser


def training_arguments():
    parser = argparse.ArgumentParser(description="CGlow trainer")
    parser.add_argument("--dataset", type=str, default='mnist', choices=DATASETS + REGRESSION_DATASETS + GRAPH_DATASETS,
                        help="Dataset to use")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--n_epoch", default=1000, type=int, help="number of epoch")
    parser.add_argument("--model", type=str, default='cglow', choices=SIMPLE_MODELS + IMAGE_MODELS + GRAPH_MODELS,
                        help="Model to use")

    parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")

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
    parser.add_argument("--mean_of_eigval", default=10, type=float, help='mean value of eigenvalues')
    parser.add_argument("--set_eigval_manually", default=None, type=str,
                        help='set the eigenvalues manually, it should be in the form of list of n_dim value e.g [50,0.003]')

    parser.add_argument("--dim_per_label", default=None, type=int, help='number of dimension to use for one class')

    parser.add_argument("--reg_use_var", action="store_true",
                        help='use variance computed wrt distance and label min dist during training')
    # METHOD SELECTION ARG
    parser.add_argument("--method", default=0, type=int, help='select between [0,1,2]')

    parser.add_argument("--restart", action="store_true",
                        help='restart from the restart.pth model')

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
