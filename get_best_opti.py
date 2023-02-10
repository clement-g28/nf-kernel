import numpy as np
import math
import torch
import glob
import os
import re

from utils.custom_glow import CGlow, WrappedModel
from utils.dataset import ImDataset, SimpleDataset, GraphDataset, RegressionGraphDataset, ClassificationGraphDataset, \
    SIMPLE_DATASETS, SIMPLE_REGRESSION_DATASETS, IMAGE_DATASETS, GRAPH_REGRESSION_DATASETS, \
    GRAPH_CLASSIFICATION_DATASETS

from utils.utils import set_seed, create_folder, initialize_gaussian_params, initialize_regression_gaussian_params, \
    save_fig, initialize_tmp_regression_gaussian_params

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from utils.testing import testing_arguments
from utils.testing import learn_or_load_modelhyperparams
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments, seqflow_arguments, graphnvp_arguments

from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model
from utils.models import IMAGE_MODELS, SIMPLE_MODELS, GRAPH_MODELS

from get_best_regression import evaluate_regression, visualize_dataset
from get_best_classification import evaluate_classification

if __name__ == '__main__':
    parser = testing_arguments()
    parser.add_argument("--method", default=0, type=int, help='select between [0,1,2]')
    parser.add_argument("--eval_type", default='regression', type=str, choices=['regression', 'classification'],
                        help='select the evaluation type')
    args = parser.parse_args()

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

    os.chdir(args.folder)
    saves = []
    for file in glob.glob("checkpoint_*/checkpoint"):
        saves.append(file)

    saves.sort()
    print(saves)

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
    # val_dataset.reduce_regression_dataset(0.5, stratified=True)

    n_dim = dataset.get_n_dim()

    if not dataset.is_regression_dataset():
        uni = np.unique(dataset.true_labels)
        dim_per_label = math.floor(n_dim / len(uni))
    else:
        dim_per_label = n_dim

    t_model_params = []
    for save_path in saves:
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
        model_loading_params = {'model': model_type, 'n_dim': n_dim, 'n_flow': n_flow,
                                'n_block': n_block, 'gaussian_params': gaussian_params, 'device': device,
                                'loading_path': save_path, 'learn_mean': learn_mean}
        if model_type == 'cglow':
            args_cglow, _ = cglow_arguments().parse_known_args()
            model_loading_params['n_channel'] = dataset.n_channel
            model_loading_params['affine'] = args_cglow.affine
            model_loading_params['conv_lu'] = not args_cglow.no_lu
        if model_type == 'ffjord':
            args_ffjord, _ = ffjord_arguments().parse_known_args()
            # args_ffjord.n_block = n_block
            model_loading_params['ffjord_args'] = args_ffjord
        elif model_type == 'moflow':
            args_moflow, _ = moflow_arguments().parse_known_args()
            model_loading_params['moflow_args'] = args_moflow

        t_model_params.append(model_loading_params)

    save_dir = './save'
    create_folder(save_dir)


    def loading_func(model, path):
        model_state, optimizer_state = torch.load(path, map_location=device)
        model.load_state_dict(model_state)
        return model


    if args.eval_type == 'regression':
        best_i = evaluate_regression(t_model_params, train_dataset, val_dataset, full_dataset=dataset,
                                     save_dir=save_dir, device=device, with_train=True, reg_use_var=reg_use_var,
                                     cus_load_func=loading_func)
    elif args.eval_type == 'classification':
        best_i = evaluate_classification(t_model_params, train_dataset, val_dataset, full_dataset=dataset,
                                         save_dir=save_dir, device=device, with_train=True, cus_load_func=loading_func)
