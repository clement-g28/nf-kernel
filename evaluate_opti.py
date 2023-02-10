import numpy as np
from PIL import Image, ImageDraw
import time
import math
import os
import re
import torch
from torchvision import utils

from utils.utils import set_seed, create_folder, save_every_pic, initialize_gaussian_params, \
    initialize_regression_gaussian_params, save_fig, initialize_tmp_regression_gaussian_params

from utils.custom_glow import CGlow, WrappedModel
from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model
from utils.models import IMAGE_MODELS, SIMPLE_MODELS, GRAPH_MODELS
from utils.dataset import ImDataset, SimpleDataset, GraphDataset, RegressionGraphDataset, ClassificationGraphDataset, \
    SIMPLE_DATASETS, SIMPLE_REGRESSION_DATASETS, IMAGE_DATASETS, GRAPH_REGRESSION_DATASETS, \
    GRAPH_CLASSIFICATION_DATASETS
from utils.density import construct_covariance
from utils.testing import learn_or_load_modelhyperparams, generate_sample, project_inZ, testing_arguments, noise_data
from utils.testing import project_between
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments
from sklearn.metrics.pairwise import rbf_kernel
from utils.graphs.kernels import compute_wl_kernel, compute_sp_kernel, compute_mslap_kernel, compute_hadcode_kernel

from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate import evaluate_regression, create_figures_XZ, evaluate_regression_preimage, \
    evaluate_regression_preimage2, evaluate_distances, evaluate_classification, generate_meanclasses, \
    test_generation_on_eigvec, evaluate_projection_1model, evaluate_interpolations

if __name__ == "__main__":
    choices = ['classification', 'projection', 'generation', 'regression']
    best_model_choices = ['classification', 'projection', 'regression']
    for choice in best_model_choices.copy():
        best_model_choices.append(choice + '_train')
    parser = testing_arguments()
    parser.add_argument('--eval_type', type=str, default='classification', choices=choices, help='evaluation type')
    parser.add_argument('--model_to_use', type=str, default='classification', choices=best_model_choices,
                        help='what best model to use for the evaluation')
    parser.add_argument("--method", default=0, type=int, help='select between [0,1,2]')
    args = parser.parse_args()
    print(args)
    set_seed(0)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]
    # DATASET #
    if dataset_name in IMAGE_DATASETS:
        dataset = ImDataset(dataset_name=dataset_name)
    elif dataset_name == 'fishtoxi':  # Special case where the data can be either graph or vectorial data
        use_graph_type = model_type in GRAPH_MODELS
        if use_graph_type:
            dataset = RegressionGraphDataset(dataset_name=dataset_name)
        else:
            dataset = SimpleDataset(dataset_name=dataset_name)
    elif dataset_name in SIMPLE_DATASETS or dataset_name in SIMPLE_REGRESSION_DATASETS:
        dataset = SimpleDataset(dataset_name=dataset_name)
    elif dataset_name in GRAPH_REGRESSION_DATASETS:
        dataset = RegressionGraphDataset(dataset_name=dataset_name)
    elif dataset_name in GRAPH_CLASSIFICATION_DATASETS:
        dataset = ClassificationGraphDataset(dataset_name=dataset_name)
    else:
        assert False, 'unknown dataset'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_path = f'{args.folder}/save/best_{args.model_to_use}_model.pth'
    assert os.path.exists(flow_path), f'snapshot path {flow_path} does not exists'
    folder_name = args.folder.split('/')[-1]

    splits = folder_name.split(',')
    for split in splits:
        if 'var' in split:
            var = float(split.split('_')[0].split('var=')[-1])

    # Retrieve parameters from name
    mean_of_eigval = var
    fixed_eigval = None
    n_block = 2  # base value
    n_flow = 32  # base value
    reg_use_var = False

    # Load the splits of the dataset used in the training phase
    train_idx_path = f'./train_idx.npy'
    val_idx_path = f'./val_idx.npy'
    if os.path.exists(train_idx_path):
        print('Loading train idx...')
        train_dataset = dataset.duplicate()
        train_dataset.load_split(train_idx_path)
    else:
        print('No train idx found, using the full dataset as train dataset...')
        train_dataset = dataset
    if args.reselect_val_idx is not None:
        train_labels = np.unique(train_dataset.true_labels)
        train_idx = train_dataset.idx
        val_dataset = dataset.duplicate()
        val_dataset.idx = np.array(
            [i for i in range(dataset.ori_X.shape[0]) if
             i not in train_idx and dataset.ori_true_labels[i] in train_labels])
        val_dataset.X = val_dataset.ori_X[val_dataset.idx]
        val_dataset.true_labels = val_dataset.ori_true_labels[val_dataset.idx]
        val_dataset.reduce_dataset('every_class', how_many=args.reselect_val_idx, reduce_from_ori=False)
    elif os.path.exists(val_idx_path):
        print('Loading val idx...')
        val_dataset = dataset.duplicate()
        val_dataset.load_split(val_idx_path)
    else:
        print('No val idx found, searching for test dataset...')
        if dataset_name == 'mnist':
            val_dataset = ImDataset(dataset_name=dataset_name, test=True)
        else:
            train_dataset, val_dataset = dataset.split_dataset(0)
        # val_dataset = ImDataset(dataset_name=dataset_name, test=True)

    # reduce train dataset size (fitting too long)
    # print('Train dataset reduced in order to accelerate. (stratified)')
    # train_dataset.reduce_regression_dataset(0.1, stratified=True)
    # val_dataset.reduce_regression_dataset(0.1, stratified=True)

    n_dim = dataset.get_n_dim()

    if not dataset.is_regression_dataset():
        uni = np.unique(dataset.true_labels)
        dim_per_label = math.floor(n_dim / len(uni))
    else:
        dim_per_label = n_dim

    # initialize gaussian params
    eigval_list = [mean_of_eigval for i in range(dim_per_label)]

    if not dataset.is_regression_dataset():
        gaussian_params = initialize_gaussian_params(dataset, eigval_list, isotrope=True,
                                                     dim_per_label=dim_per_label, fixed_eigval=fixed_eigval)
    else:
        gaussian_params = initialize_regression_gaussian_params(dataset, eigval_list,
                                                                isotrope=True,
                                                                dim_per_label=dim_per_label,
                                                                fixed_eigval=fixed_eigval)

    learn_mean = True
    if model_type == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        n_channel = dataset.n_channel
        affine = args_cglow.affine
        no_lu = args_cglow.no_lu
        # Load model
        model_single = CGlow(n_channel, n_flow, n_block, affine=affine, conv_lu=not no_lu,
                             gaussian_params=gaussian_params, device=device, learn_mean='lmean' in folder_name)
    elif model_type == 'seqflow':
        model_single = load_seqflow_model(n_dim, n_flow, gaussian_params=gaussian_params,
                                          learn_mean=learn_mean, reg_use_var=reg_use_var, dataset=dataset)

    elif model_type == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        args_ffjord.n_block = n_block
        model_single = load_ffjord_model(args_ffjord, n_dim, gaussian_params=gaussian_params,
                                         learn_mean=learn_mean, reg_use_var=reg_use_var, dataset=dataset)
    elif model_type == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        args_moflow.noise_scale = 0
        model_single = load_moflow_model(args_moflow,
                                         gaussian_params=gaussian_params,
                                         learn_mean=learn_mean, reg_use_var=reg_use_var,
                                         dataset=dataset)
    else:
        assert False, 'unknown model type!'

    model_state = torch.load(flow_path, map_location=device)
    model_single.load_state_dict(model_state)
    model = model_single

    model = model.to(device)
    model.eval()

    os.chdir(args.folder)

    save_dir = './save'
    create_folder(save_dir)

    eval_type = args.eval_type
    if eval_type == 'classification':
        dataset_name_eval = ['mnist', 'double_moon', 'iris', 'bcancer'] + GRAPH_CLASSIFICATION_DATASETS
        assert dataset_name in dataset_name_eval, f'Classification can only be evaluated on {dataset_name_eval}'
        evaluate_classification(model, train_dataset, val_dataset, save_dir, device)
    elif eval_type == 'generation':
        dataset_name_eval = ['mnist']
        assert dataset_name in dataset_name_eval, f'Generation can only be evaluated on {dataset_name_eval}'
        # GENERATION
        how_much = [1, 10, 30, 50, 78]
        img_size = dataset.im_size
        z_shape = model_single.calc_last_z_shape(img_size)
        for n in how_much:
            test_generation_on_eigvec(model, gaussian_params=gaussian_params, z_shape=z_shape, how_much_dim=n,
                                      device=device, sample_per_label=10, save_dir=save_dir)
        generate_meanclasses(model, train_dataset, device)
    elif eval_type == 'projection':
        img_size = dataset.im_size
        z_shape = model_single.calc_last_z_shape(img_size)
        # PROJECTIONS
        if dataset_name == 'mnist':
            noise_types = ['gaussian', 'speckle', 'poisson', 's&p']
            for noise_type in noise_types:
                evaluate_projection_1model(model, train_dataset, val_dataset, gaussian_params=gaussian_params,
                                           z_shape=z_shape, how_much=dim_per_label, device=device, save_dir=save_dir,
                                           proj_type='zpca_l', noise_type=noise_type, eval_gaussian_std=.2)
        elif dataset_name in ['single_moon', 'double_moon']:
            noise_type = 'gaussian'
            n_principal_dim = np.count_nonzero(gaussian_params[0][-2] > 1)
            evaluate_distances(model, train_dataset, val_dataset, gaussian_params=gaussian_params,
                               z_shape=z_shape, how_much=n_principal_dim, kpca_types=['linear', 'rbf', 'poly'],
                               device=device, save_dir=save_dir, proj_type='zpca_l', noise_type=noise_type,
                               eval_gaussian_std=.1)
        else:
            dataset_name_eval = ['mnist', 'single_moon', 'double_moon']
            assert dataset_name in dataset_name_eval, f'Projection can only be evaluated on {dataset_name_eval}'
    elif eval_type == 'regression':
        assert dataset.is_regression_dataset(), 'the dataset is not made for regression purposes'
        evaluate_regression(model, train_dataset, val_dataset, save_dir, device)
        _, Z = create_figures_XZ(model, train_dataset, save_dir, device, std_noise=0.1,
                                 only_Z=isinstance(dataset, GraphDataset))
        evaluate_regression_preimage(model, val_dataset, device, save_dir, print_as_mol=True, print_as_graph=True)
        evaluate_regression_preimage2(model, val_dataset, device, save_dir, n_y=20, n_samples_by_y=10,
                                      print_as_mol=True, print_as_graph=True)
        evaluate_interpolations(model, val_dataset, device, save_dir, n_sample=100, n_interpolation=30, Z=Z,
                                print_as_mol=True, print_as_graph=True)
