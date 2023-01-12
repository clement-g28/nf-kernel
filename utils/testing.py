import os
import pickle
import argparse
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from skimage.util import random_noise

from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge


def testing_arguments():
    parser = argparse.ArgumentParser(description="CGlow testing")
    # parser.add_argument("--dataset", type=str, default='mnist', help="Dataset to use")
    # parser.add_argument("--n_flow", default=32, type=int, help="number of flows in each block")
    # parser.add_argument("--n_block", default=2, type=int, help="number of blocks")

    # CGLOW parameters
    parser.add_argument(
        "--no_lu",
        action="store_true",
        help="use plain convolution instead of LU decomposed version",
    )
    parser.add_argument("--affine", action="store_true", help="use affine coupling instead of additive")

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
