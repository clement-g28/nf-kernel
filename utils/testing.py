import os
import pickle
import argparse
import numpy as np
import re
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from skimage.util import random_noise

from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from utils.utils import initialize_class_gaussian_params, initialize_regression_gaussian_params, \
    initialize_tmp_regression_gaussian_params
from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model, load_cglow_model
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments, seqflow_arguments, graphnvp_arguments


def testing_arguments():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--n_bits", default=5, type=int, help="number of bits used only with image dataset")

    parser.add_argument('--folder', type=str, default='./checkpoint')
    parser.add_argument('--reselect_val_idx', type=int, default=None,
                        help='Reselect a set of validation data from the entire dataset except data from training')
    return parser


def learn_or_load_modelhyperparams(X_train, tlabels, kernel_name, param_grid, save_dir, model_type, scaler=False,
                                   scorer=None, save=True, force_train=False):
    # Example of use : pca = learn_or_load_modelhyperparams(X_lab, None, kernel_name, param_gridlin, save_dir,
    # model_type=('PCA', KernelPCA()), scaler=scaler, scorer=my_scorer)
    # model_type should be ('name', model)
    if not os.path.exists(f'{save_dir}/params_{model_type[0]}_{kernel_name}.pkl') or force_train:
        pipeline_li = [('scaler', StandardScaler())] if scaler else []
        pipeline_li += [model_type]
        pipeline_linsvc = Pipeline(pipeline_li)
        model = GridSearchCV(pipeline_linsvc, param_grid, n_jobs=4, verbose=1, scoring=scorer)
        if tlabels is not None:
            model.fit(X_train, tlabels)
        else:
            model.fit(X_train)
        keys = [k.split('__')[-1] for k, v in param_grid[0].items()]
        ind_step = 0 if not scaler else 1
        learned_params = {k: eval(f'{str(model.best_estimator_.steps[ind_step][1])}.{k}') for k in keys}
        # kernel_name = learned_params['kernel']
        print(f'Learned hyperparams : {learned_params}')
        if save:
            with open(f'{save_dir}/params_{model_type[0]}_{kernel_name}.pkl', 'wb') as f:
                pickle.dump(learned_params, f)
    else:
        with open(f'{save_dir}/params_{model_type[0]}_{kernel_name}.pkl', 'rb') as f:
            learned_params = pickle.load(f)
        print(f'Loaded hyperparams : {learned_params}')
        if scaler:
            model = make_pipeline(StandardScaler(), type(model_type[-1])(**learned_params))
        else:
            model = type(model_type[-1])(**learned_params)
        if tlabels is not None:
            model.fit(X_train, tlabels)
        else:
            model.fit(X_train)
    print(f'Fitting done.')
    return model


def save_modelhyperparams(model, param_grid, save_dir, model_type, kernel_name, scaler_used=False):
    keys = [k.split('__')[-1] for k, v in param_grid[0].items()]
    ind_step = 0 if not scaler_used else 1
    learned_params = {k: eval(f'{str(model.best_estimator_.steps[ind_step][1])}.{k}') for k in keys}
    with open(f'{save_dir}/params_{model_type[0]}_{kernel_name}.pkl', 'wb') as f:
        pickle.dump(learned_params, f)
    print(f'Model saved.')


def generate_sample(mean, covariance, nb_sample):
    return np.random.multivariate_normal(mean, covariance, nb_sample)


def project_inZ(z_noisy, params, how_much):
    means, gp = params
    eigenvec = gp[0]
    eigenval = gp[1]
    std = np.sqrt(eigenval)
    noisy_z_norm = (z_noisy - means) / std
    indexes = np.argsort(-eigenval, kind='mergesort')
    proj_z_noisy = noisy_z_norm @ eigenvec.T

    k = how_much
    proj = proj_z_noisy[:, :k] @ eigenvec[indexes[:k]]
    proj *= std
    proj += means

    return proj


def project_between(z, a, b):
    assert len(z.shape) > 1, 'z should be of shape (batch, -1)'
    direction = (a - b).astype(np.float)
    magnitude = np.linalg.norm(a - b)
    direction = direction / magnitude
    z_norm = np.expand_dims((z - b) / magnitude, 0)
    direction = np.expand_dims(direction, 0)
    dot_val = z_norm @ direction.T
    proj = (dot_val @ direction).squeeze()

    proj *= magnitude
    proj += b
    return proj, dot_val


def noise_data(data, noise_type, gaussian_std, gaussian_mean, clip=True):
    if noise_type == 'gaussian':
        noisy = random_noise(data, mode='gaussian', mean=gaussian_mean, var=gaussian_std * gaussian_std, clip=clip)
    elif noise_type == 'poisson':
        noisy = random_noise(data, mode='poisson', clip=clip)
    elif noise_type == 'speckle':
        noisy = random_noise(data, mode='speckle', mean=gaussian_mean, var=gaussian_std * gaussian_std, clip=clip)
    elif noise_type == 's&p':
        noisy = random_noise(data, mode='s&p', clip=clip)
    else:
        assert False, 'unknown noise type'
    return noisy


import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_fig(X, reconstructed_X, labels, label_max, save_path, limits=None, size=7, noisy_X=None):
    plt.scatter(X[:, 0], X[:, 1], s=size, c=labels, zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")

    c_labels = np.zeros(reconstructed_X.shape[0])
    c_labels[:reconstructed_X.shape[0]] = label_max + 1
    plt.scatter(reconstructed_X[:, 0], reconstructed_X[:, 1], s=size, c='r', zorder=2)
    plt.xlabel("x")
    plt.ylabel("y")

    if noisy_X is not None:
        plt.scatter(noisy_X[:, 0], noisy_X[:, 1], s=size, c=labels, zorder=3)
        plt.xlabel("x")
        plt.ylabel("y")

    # Links
    to_show = np.zeros((2 * X.shape[0], X.shape[-1]))
    print(to_show.shape)
    for i in range(0, X.shape[0]):
        to_show[2 * i, :] = X[i, :]
        to_show[2 * i + 1, :] = reconstructed_X[i, :]

    for i in range(0, X.shape[0]):
        plt.plot(to_show[2 * i:2 * i + 2, 0], to_show[2 * i:2 * i + 2, 1], color='lime', zorder=1)

    if limits:
        x_xmin, x_xmax, x_ymin, x_ymax = limits
        plt.xlim([x_xmin - 1, x_xmax + 1])
        plt.ylim([x_ymin - 1, x_ymax + 1])

    plt.savefig(fname=f'{save_path}.png', format='png')
    # Grid
    # plt.scatter(z_grid_inX[:, 0], z_grid_inX[:, 1], c=color, s=5, alpha=0.3)
    plt.close()


def retrieve_params_from_name(folder_name):
    # Retrieve parameters from name
    splits = folder_name.split('_')
    label = None
    uniform_eigval = False
    mean_of_eigval = None
    gaussian_eigval = None
    dim_per_label = None
    fixed_eigval = None
    n_block = 2  # base value
    n_flow = 32  # base value
    reg_use_var = False
    split_graph_dim = False

    for split in splits:
        b = re.search("^b\d{1,2}$", split)
        f = re.search("^f\d{1,3}", split)
        if b is not None:
            n_block = int(b.string.replace('b', ''))
        elif f is not None:
            n_flow = int(f.string.replace('f', ''))
        elif 'label' in split:
            label_split = split
            label = int(label_split.replace("label", ""))
            print(f'Flow trained with reduces dataset to one label: {label}')
        elif 'manualeigval' in split:
            manualeigval_split = split
            fixed_eigval = list(map(float, manualeigval_split.replace("manualeigval", "").split('-')))
            print(f'Flow trained with fixed eigenvalues: {fixed_eigval}')
        elif 'eigvaluniform' in split:
            uniform_eigval = True
            uniform_split = split
            mean_of_eigval = uniform_split.replace("eigvaluniform", "")
            if 'e' not in mean_of_eigval:
                mean_of_eigval = float(mean_of_eigval.replace("-", "."))
            else:
                mean_of_eigval = float(mean_of_eigval)
            print(f'Flow trained with uniform eigenvalues: {mean_of_eigval}')
        elif 'eigvalgaussian' in split:
            gaussian_split = split
            in_split = gaussian_split.split("std")
            mean_of_eigval = in_split[0].replace("eigvalgaussian", "")
            mean_of_eigval = float(mean_of_eigval.replace("-", "."))
            std_value = float(str(in_split[-1]).replace('-', '.'))
            gaussian_eigval = [0.0, std_value]
            print(f'Flow trained with gaussian eigenvalues params: {mean_of_eigval},{gaussian_eigval}')
        elif 'dimperlab' in split:
            dpl_split = split
            dim_per_label = int(dpl_split.replace("dimperlab", ""))
            print(f'Flow trained with dimperlab: {dim_per_label}')
        elif 'usevar' in split:
            reg_use_var = True
            print(f'Flow trained using variance.')
        elif 'splitgraphdim' in split:
            split_graph_dim = True
            print(f'Flow trained using variance.')

    return n_block, n_flow, mean_of_eigval, dim_per_label, label, fixed_eigval, uniform_eigval, gaussian_eigval, reg_use_var, split_graph_dim


def initialize_gaussian_params(args, dataset, fixed_eigval, uniform_eigval, gaussian_eigval, mean_of_eigval,
                               dim_per_label, isotrope, split_graph_dim):
    # initialize gaussian params
    if fixed_eigval is None:
        if uniform_eigval:
            eigval_list = [mean_of_eigval for i in range(dim_per_label)]
        elif gaussian_eigval is not None:
            import scipy.stats as st

            dist = st.norm(loc=gaussian_eigval[0], scale=gaussian_eigval[1])
            border = 1.6
            step = border * 2 / dim_per_label
            x = np.linspace(-border, border, dim_per_label) if (dim_per_label % 2) != 0 else np.concatenate(
                (np.linspace(-border, gaussian_eigval[0], int(dim_per_label / 2))[:-1], [gaussian_eigval[0]],
                 np.linspace(step, border, int(dim_per_label / 2))))
            eigval_list = dist.pdf(x)
            mean_eigval = mean_of_eigval
            a = mean_eigval * dim_per_label / eigval_list.sum()
            eigval_list = a * eigval_list
            eigval_list[np.where(eigval_list < 1)] = 1
        else:
            assert False, 'Unknown distribution !'
    else:
        eigval_list = None

    if not dataset.is_regression_dataset():
        gaussian_params = initialize_class_gaussian_params(dataset, eigval_list, isotrope=isotrope,
                                                           dim_per_label=dim_per_label, fixed_eigval=fixed_eigval,
                                                           split_graph_dim=split_graph_dim)
    else:
        if args.method == 0:
            gaussian_params = initialize_regression_gaussian_params(dataset, eigval_list,
                                                                    isotrope=isotrope,
                                                                    dim_per_label=dim_per_label,
                                                                    fixed_eigval=fixed_eigval)
        elif args.method == 1:
            gaussian_params = initialize_tmp_regression_gaussian_params(dataset, eigval_list)
        elif args.method == 2:
            gaussian_params = initialize_tmp_regression_gaussian_params(dataset, eigval_list, ones=True)
        else:
            assert False, 'no method selected'
    return gaussian_params


def prepare_model_loading_params(model_loading_params, dataset, model_type):
    if model_type == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        args_cglow.n_flow = model_loading_params['n_flow']
        model_loading_params['cglow_args'] = args_cglow
        model_loading_params['n_channel'] = dataset.n_channel
    if model_type == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        args_ffjord.n_block = model_loading_params['n_block']
        model_loading_params['ffjord_args'] = args_ffjord
    elif model_type == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        model_loading_params['moflow_args'] = args_moflow
    return model_loading_params


def load_model_from_params(model_loading_params, dataset):
    if model_loading_params['model'] == 'cglow':
        model_single = load_cglow_model(model_loading_params['cglow_args'], model_loading_params['n_channel'],
                                        gaussian_params=model_loading_params['gaussian_params'],
                                        learn_mean=model_loading_params['learn_mean'],
                                        device=model_loading_params['device'])
    elif model_loading_params['model'] == 'seqflow':
        model_single = load_seqflow_model(model_loading_params['n_dim'], model_loading_params['n_flow'],
                                          gaussian_params=model_loading_params['gaussian_params'],
                                          learn_mean=model_loading_params['learn_mean'], dataset=dataset)

    elif model_loading_params['model'] == 'ffjord':
        model_single = load_ffjord_model(model_loading_params['ffjord_args'], model_loading_params['n_dim'],
                                         gaussian_params=model_loading_params['gaussian_params'],
                                         learn_mean=model_loading_params['learn_mean'], dataset=dataset)
    elif model_loading_params['model'] == 'moflow':
        model_single = load_moflow_model(model_loading_params['moflow_args'],
                                         gaussian_params=model_loading_params['gaussian_params'],
                                         learn_mean=model_loading_params['learn_mean'],
                                         dataset=dataset)
    else:
        assert False, 'unknown model type!'
    return model_single


def load_split_dataset(dataset, train_idx_path, val_idx_path, reselect_val_idx=None):
    if os.path.exists(train_idx_path):
        print('Loading train idx...')
        train_dataset = dataset.duplicate()
        train_dataset.load_split(train_idx_path)
    else:
        print('No train idx found, using the full dataset as train dataset...')
        train_dataset = dataset
    if reselect_val_idx is not None:
        train_labels = np.unique(train_dataset.true_labels)
        train_idx = train_dataset.idx
        val_dataset = dataset.duplicate()
        val_dataset.idx = np.array(
            [i for i in range(dataset.ori_X.shape[0]) if
             i not in train_idx and dataset.ori_true_labels[i] in train_labels])
        val_dataset.X = val_dataset.ori_X[val_dataset.idx]
        val_dataset.true_labels = val_dataset.ori_true_labels[val_dataset.idx]
        val_dataset.reduce_dataset('every_class', how_many=reselect_val_idx, reduce_from_ori=False)
    elif os.path.exists(val_idx_path):
        print('Loading val idx...')
        val_dataset = dataset.duplicate()
        val_dataset.load_split(val_idx_path)
    else:
        print('No val idx found, searching for test dataset...')
        train_dataset, val_dataset = dataset.split_dataset(0)
    return train_dataset, val_dataset
