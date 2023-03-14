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

from utils.utils import set_seed, create_folder, initialize_class_gaussian_params, initialize_regression_gaussian_params, \
    save_fig, initialize_tmp_regression_gaussian_params

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from utils.testing import testing_arguments
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments, seqflow_arguments, graphnvp_arguments

from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model
from utils.models import IMAGE_MODELS, SIMPLE_MODELS, GRAPH_MODELS

from get_best_regression import evaluate_regression, visualize_dataset
from get_best_classification import evaluate_classification

from utils.testing import retrieve_params_from_name, learn_or_load_modelhyperparams, initialize_gaussian_params, \
    prepare_model_loading_params, load_split_dataset, load_model_from_params
from utils.utils import load_dataset

if __name__ == '__main__':
    parser = testing_arguments()
    parser.add_argument("--method", default=0, type=int, help='select between [0,1,2]')
    parser.add_argument("--eval_type", default='regression', type=str, choices=['regression', 'classification'],
                        help='select the evaluation type')
    args = parser.parse_args()

    set_seed(0)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]
    # DATASET #
    dataset = load_dataset(args, dataset_name, model_type, to_evaluate=True)

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
    train_dataset, val_dataset = load_split_dataset(dataset, train_idx_path, val_idx_path,
                                                    reselect_val_idx=args.reselect_val_idx)

    # reduce train dataset size (fitting too long)
    # print('Train dataset reduced in order to accelerate. (stratified)')
    # train_dataset.reduce_dataset_ratio(0.5, stratified=True)
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
            gaussian_params = initialize_class_gaussian_params(dataset, eigval_list, isotrope=True,
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
        t_model_params.append(prepare_model_loading_params(model_loading_params, dataset, model_type))

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
