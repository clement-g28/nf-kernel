import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import utils

from utils.custom_glow import WrappedModel
from utils.dataset import GraphDataset, ImDataset, GRAPH_CLASSIFICATION_DATASETS
from utils.density import construct_covariance
from utils.utils import format, organise_data, predict_by_log_p_with_gaussian_params
from utils.graphs.graph_utils import save_nx_graph, save_nx_graph_attr
from utils.graphs.kernels import compute_wl_kernel, compute_sp_kernel, compute_mslap_kernel, compute_hadcode_kernel, \
    compute_propagation_kernel
from utils.graphs.mol_utils import valid_mol, construct_mol
from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model, load_cglow_model
from utils.testing import generate_sample, project_inZ, testing_arguments, noise_data, project_between, \
    retrieve_params_from_name, learn_or_load_modelhyperparams, initialize_gaussian_params, \
    load_split_dataset
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments
from utils.utils import set_seed, create_folder, save_every_pic, save_fig, save_projection_fig, load_dataset

from scipy.stats import kruskal

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from ordered_set import OrderedSet


def classification_score(pred, true):
    return np.count_nonzero((pred == true)) / true.shape[0]


def test_generation_on_eigvec(model_single, val_dataset, gaussian_params, z_shape, how_much_dim, device,
                              sample_per_label=10, save_dir='./save', debug=False):
    create_folder(f'{save_dir}/test_generation')

    all_generation = []
    n_image_per_lab = 10
    for i, gaussian_param in enumerate(gaussian_params):
        mean = model_single.means[i].detach().cpu().numpy().squeeze()
        eigenvec = gaussian_param[1]
        eigenval = gaussian_param[2]
        indexes = np.argsort(-eigenval, kind='mergesort')
        # indexes = np.flip(np.argsort(eigenval))
        indexes = indexes[:how_much_dim]
        vecs = eigenvec[indexes][:, indexes]
        vals = eigenval[indexes]
        cov = construct_covariance(vecs, vals)
        i_mean = mean[indexes]
        z = generate_sample(i_mean, cov, nb_sample=sample_per_label)
        res = np.zeros((sample_per_label, mean.shape[0]))
        res[:, indexes] = z
        other_indexes = [ind for ind in range(0, mean.shape[0]) if ind not in indexes]
        res[:, other_indexes] = mean[other_indexes]

        z_sample = torch.from_numpy(res).reshape(sample_per_label, *z_shape).float().to(device)

        if debug:
            create_folder(f"{save_dir}/test_generation/eigvec_flows_gen/{str(i)}_{how_much_dim}")
            images = model_single.reverse_debug(z_sample, val_dataset,
                                                save_dir=f"{save_dir}/test_generation/eigvec_flows_gen/{str(i)}_{how_much_dim}").cpu().data
        else:
            images = model_single.reverse(z_sample).cpu().data
        images = val_dataset.rescale(images)
        utils.save_image(
            images,
            f"{save_dir}/test_generation/label{str(i)}_{how_much_dim}dim.png",
            normalize=True,
            nrow=7,
            range=(0, 255),
        )
        all_generation.append(images[:n_image_per_lab])

    # mean_images = generate_meanclasses(model_single, val_dataset, device, save_dir)
    # all_with_means = []
    # for i, n_generation in enumerate(all_generation):
    #     all_with_means.append(n_generation)
    #     all_with_means.append(np.expand_dims(mean_images[i], axis=0))
    all_generation = np.concatenate(all_generation, axis=0)

    utils.save_image(
        torch.from_numpy(all_generation),
        f"{save_dir}/test_generation/all_label_{how_much_dim}dim.png",
        normalize=True,
        nrow=10,
        range=(0, 255),
    )

    # all_with_means = np.concatenate(all_with_means, axis=0)
    #
    # methods = ([str(i) for i in range(0, n_image_per_lab)] + ['mean']) * len(gaussian_params)
    # labels = [[g[-1]] * (n_image_per_lab + 1) for g in gaussian_params]
    # labels = [item for sublist in labels for item in sublist]
    # save_every_pic(f'{save_dir}/test_generation/every_pics/{how_much_dim}', all_with_means, methods, labels,
    #                rgb=val_dataset.n_channel > 1)


def evaluate_projection_1model(model, train_dataset, val_dataset, gaussian_params, z_shape, how_much, device,
                               save_dir='./save', proj_type='gp', noise_type='gaussian', eval_gaussian_std=.2,
                               batch_size=20, how_much_per_lab=6):
    train_dataset.ori_X = train_dataset.X
    val_dataset.ori_X = val_dataset.X
    train_dataset.ori_true_labels = train_dataset.true_labels
    val_dataset.ori_true_labels = val_dataset.true_labels
    train_dataset.idx = np.array([i for i in range(train_dataset.X.shape[0])])
    val_dataset.idx = np.array([i for i in range(val_dataset.X.shape[0])])

    kpca_types = ['rbf', 'poly']  # best classifiers
    grid_im, distances_results = projection_evaluation(model, train_dataset, val_dataset, gaussian_params,
                                                       z_shape, how_much, kpca_types, device, save_dir=save_dir,
                                                       proj_type=proj_type, noise_type=noise_type,
                                                       eval_gaussian_std=eval_gaussian_std, batch_size=batch_size,
                                                       how_much_per_lab=how_much_per_lab)

    grid_im = np.concatenate(grid_im, axis=0)
    grid_im = val_dataset.rescale(grid_im)

    # used_method = ['ori', 'noisy', 'linear'] + kpca_types + [proj_type]
    used_method = ['ori', 'noisy'] + kpca_types + [proj_type]
    methods = used_method * len(gaussian_params) * how_much_per_lab
    labels = [[str(g[-1]) + f'_{n}'] * len(used_method) for g in gaussian_params for n in range(how_much_per_lab)]
    labels = [item for sublist in labels for item in sublist]
    save_every_pic(f'{save_dir}/projections/{noise_type}/every_pics/{proj_type}', grid_im, methods, labels,
                   add_str=proj_type, rgb=val_dataset.n_channel > 1)

    nrow = math.floor(grid_im.shape[0] / (np.unique(val_dataset.true_labels).shape[0] * how_much_per_lab))
    utils.save_image(
        torch.from_numpy(grid_im),
        f"{save_dir}/projections/{noise_type}/projection_eval_{proj_type}.png",
        normalize=True,
        nrow=nrow,
        range=(0, 255),
    )


def projection_evaluation(model, train_dataset, val_dataset, gaussian_params, z_shape, how_much, kpca_types, device,
                          save_dir='./save', proj_type='gp', noise_type='gaussian', eval_gaussian_std=.2,
                          batch_size=20, how_much_per_lab=6):
    create_folder(f'{save_dir}/projections')

    if train_dataset.dataset_name != 'olivetti_faces':
        train_noised = noise_data(train_dataset.X / 255, noise_type=noise_type, gaussian_mean=0,
                                  gaussian_std=eval_gaussian_std)
        train_dataset_noised = train_dataset.duplicate()
        train_dataset_noised.X = (train_noised * 255).astype(np.uint8)
        train_normalized_noised = (train_noised - train_dataset.norm_mean) / train_dataset.norm_std
    else:
        train_noised = noise_data(train_dataset.X, noise_type=noise_type, gaussian_mean=0,
                                  gaussian_std=eval_gaussian_std)
        train_dataset_noised = train_dataset.duplicate()
        train_dataset_noised.X = (train_noised).astype(np.uint8)
        train_normalized_noised = (train_noised - train_dataset.norm_mean) / train_dataset.norm_std

    loader = train_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    if proj_type in ['zpca', 'zpca_l']:
        Z = []
        tlabels = []
        with torch.no_grad():
            for j, data in enumerate(loader):
                inp, labels = data
                inp = train_dataset.format_data(inp, device)
                labels = labels.to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                Z.append(out.detach().cpu().numpy())
                tlabels.append(labels.detach().cpu().numpy())
        Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)
        tlabels = np.concatenate(tlabels, axis=0)

    if val_dataset.dataset_name != 'olivetti_faces':
        val_noised = noise_data(val_dataset.X / 255, noise_type=noise_type, gaussian_mean=0,
                                gaussian_std=eval_gaussian_std)
        val_dataset_noised = val_dataset.duplicate()
        val_dataset_noised.X = (val_noised * 255).astype(np.uint8)
        val_normalized_noised = (val_noised - train_dataset.norm_mean) / train_dataset.norm_std
    else:
        val_noised = noise_data(val_dataset.X, noise_type=noise_type, gaussian_mean=0,
                                gaussian_std=eval_gaussian_std)
        val_dataset_noised = val_dataset.duplicate()
        val_dataset_noised.X = (val_noised).astype(np.uint8)
        val_normalized_noised = (val_noised - train_dataset.norm_mean) / train_dataset.norm_std

    val_loader = val_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    val_inZ = []
    elabels = []
    with torch.no_grad():
        for j, data in enumerate(val_loader):
            inp, labels = data
            inp = train_dataset.format_data(inp, device)
            labels = labels.to(device)
            log_p, distloss, logdet, out = model(inp, labels)
            val_inZ.append(out.detach().cpu().numpy())
            elabels.append(labels.detach().cpu().numpy())
    val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(val_dataset), -1)
    elabels = np.concatenate(elabels, axis=0)

    if val_dataset.dataset_name != 'olivetti_faces':
        val_normalized = (val_dataset.X / 255 - val_dataset.norm_mean) / val_dataset.norm_std
    else:
        val_normalized = (val_dataset.X - val_dataset.norm_mean) / val_dataset.norm_std

    # ZPCA
    if proj_type == 'zpca':
        # KPCA
        kpca_projs = []
        for j, kpca_type in enumerate(kpca_types):
            print(f'{kpca_type}-PCA processing...')
            kpca = KernelPCA(n_components=how_much, kernel=kpca_type, fit_inverse_transform=True)
            kpca.fit(train_normalized_noised.reshape(train_normalized_noised.shape[0], -1))

            kpca_projection = kpca.transform(val_normalized_noised.reshape(val_normalized_noised.shape[0], -1))
            kpca_reconstruct = kpca.inverse_transform(kpca_projection)
            kpca_projs.append(kpca_reconstruct)

        # ZPCA
        zpca = PCA(n_components=how_much)
        zpca.fit(Z)
        pca_projection = zpca.transform(val_inZ)
        proj = zpca.inverse_transform(pca_projection)
        ordered_val = val_normalized
        ordered_elabels = elabels
    else:
        kpca_projs = []
        for type in kpca_types:
            kpca_projs.append([])
        projs = []
        ordered_val = []
        ordered_elabels = []
        for i, gaussian_param in enumerate(gaussian_params):
            gmean = model.means[i].detach().cpu().numpy()
            gp = gaussian_param[1:-1]
            label = gaussian_param[-1]
            if label in elabels:

                train_normalized_noised_lab = train_normalized_noised[np.where(train_dataset.true_labels == label)]
                indexes = np.where(elabels == label)[0]
                val_Z_lab = val_inZ[indexes]
                val_normalized_lab = val_normalized[indexes]
                val_normalized_noised_lab = val_normalized_noised[indexes]

                ordered_elabels.append(np.array([label for _ in range(indexes.shape[0])]))

                # K-PCA
                for j, kpca_type in enumerate(kpca_types):
                    print(f'{kpca_type}-PCA (label {label}) processing...')
                    kpca = KernelPCA(n_components=how_much, kernel=kpca_type, fit_inverse_transform=True)
                    kpca.fit(train_normalized_noised_lab.reshape(train_normalized_noised_lab.shape[0], -1))

                    kpca_projection = kpca.transform(
                        val_normalized_noised_lab.reshape(val_normalized_noised_lab.shape[0], -1))
                    kpca_reconstruct = kpca.inverse_transform(kpca_projection)
                    kpca_projs[j].append(kpca_reconstruct)

                if proj_type == 'zpca_l':
                    # Z-PCA
                    Z_lab = Z[np.where(tlabels == label)]
                    pca = PCA(n_components=how_much)
                    pca.fit(Z_lab)
                    pca_projection = pca.transform(val_Z_lab)
                    proj = pca.inverse_transform(pca_projection)
                else:
                    proj = project_inZ(val_Z_lab, params=(gmean, gp), how_much=how_much)
                projs.append(proj)
                ordered_val.append(val_normalized_lab)
        ordered_elabels = np.concatenate(ordered_elabels, axis=0)
        ordered_val = np.concatenate(ordered_val, axis=0)
        for k, kpca_proj in enumerate(kpca_projs):
            kpca_projs[k] = np.concatenate(kpca_proj, axis=0)
        proj = np.concatenate(projs, axis=0)

    distances_results = {}
    # KPCA results
    for j, kpca_type in enumerate(kpca_types):
        mean_dist = np.mean(np.sum(np.abs(kpca_projs[j] - ordered_val.reshape(ordered_val.shape[0], -1)), axis=1))
        print(f'{kpca_type}-PCA: {mean_dist}')
        distances_results[f'{kpca_type}-PCA'] = mean_dist

    # Our approach results
    nb_batch = math.ceil(proj.shape[0] / batch_size)
    all_im = []
    proj = torch.from_numpy(proj)
    for j in range(nb_batch):
        size = proj[j * batch_size:(j + 1) * batch_size].shape[0]
        z_b = proj[j * batch_size:(j + 1) * batch_size].reshape(size, *z_shape).float().to(device)

        images = model.reverse(z_b).cpu().data.numpy()
        all_im.append(images)
    all_im = np.concatenate(all_im, axis=0)

    our_dist = np.sum(np.abs(all_im.reshape(all_im.shape[0], -1) - ordered_val.reshape(ordered_val.shape[0], -1)),
                      axis=1)
    mean_dist = np.mean(our_dist)
    print(f'Our approach {proj_type} : {mean_dist}')
    distances_results[f'Our-{proj_type}'] = mean_dist

    # Save one of each
    grid_im = []
    vis_index = 0
    for i, gaussian_param in enumerate(gaussian_params):
        label = gaussian_param[-1]
        if label in elabels:
            ind = np.where(ordered_elabels == label)[0]
            for n in range(how_much_per_lab):
                grid_im.append(
                    np.expand_dims(val_normalized[np.where(val_dataset.true_labels == label)][vis_index + n], axis=0))
                grid_im.append(
                    np.expand_dims(val_normalized_noised[np.where(val_dataset.true_labels == label)][vis_index + n],
                                   axis=0))
                # grid_im.append(np.expand_dims(pca_projs[ind][vis_index].reshape(val_dataset.X[0].shape), axis=0))
                for kpca_proj in kpca_projs:
                    grid_im.append(
                        np.expand_dims(kpca_proj[ind][vis_index + n].reshape(val_dataset.X[0].shape), axis=0))
                grid_im.append(np.expand_dims(all_im[ind][vis_index + n], axis=0))

    return grid_im, distances_results


def compression_evaluation(model, train_dataset, val_dataset, gaussian_params, z_shape, how_much, device,
                           save_dir='./save', proj_type='gp', noise_type='gaussian', eval_gaussian_std=.2,
                           batch_size=400, how_much_by_label=6):
    assert isinstance(how_much, list), 'how-much should be a list of number of dimension to project on'
    create_folder(f'{save_dir}/projections')

    if train_dataset.dataset_name != 'olivetti_faces':
        train_noised = noise_data(train_dataset.X / 255, noise_type=noise_type, gaussian_mean=0,
                                  gaussian_std=eval_gaussian_std)
        train_dataset_noised = train_dataset.duplicate()
        train_dataset_noised.X = (train_noised * 255).astype(np.uint8)
        # train_normalized_noised = (train_noised - train_dataset.norm_mean) / train_dataset.norm_std
    else:
        train_noised = noise_data(train_dataset.X, noise_type=noise_type, gaussian_mean=0,
                                  gaussian_std=eval_gaussian_std)
        train_dataset_noised = train_dataset.duplicate()
        train_dataset_noised.X = train_noised

    loader = train_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    if proj_type in ['zpca', 'zpca_l']:
        print(f'Z train generation...')
        Z = []
        tlabels = []
        with torch.no_grad():
            for j, data in enumerate(loader):
                inp, labels = data
                inp = train_dataset.format_data(inp, device)
                labels = labels.to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                Z.append(out.detach().cpu().numpy())
                tlabels.append(labels.detach().cpu().numpy())
        Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)
        tlabels = np.concatenate(tlabels, axis=0)

    done = []
    val_data = []
    elabels = []
    for i, gaussian_param in enumerate(gaussian_params):
        label = gaussian_param[-1]
        if label in val_dataset.true_labels:
            idx = np.where(val_dataset.true_labels == label)[0]
            for n in range(how_much_by_label):
                rand_i = np.random.choice(idx)
                while rand_i in done:
                    rand_i = np.random.choice(idx)
                done.append(rand_i)

                val_data.append(np.expand_dims(val_dataset.X[rand_i], axis=0))
                elabels.append(label)
    val_data = np.concatenate(val_data, axis=0)
    elabels = np.array(elabels)

    if train_dataset.dataset_name != 'olivetti_faces':
        val_noised = noise_data(val_data / 255, noise_type=noise_type, gaussian_mean=0, gaussian_std=eval_gaussian_std)
        val_normalized_noised = ((val_noised - val_dataset.norm_mean) / val_dataset.norm_std).astype(np.float32)
        val_normalized = ((val_data / 255 - val_dataset.norm_mean) / val_dataset.norm_std).astype(np.float32)
    else:
        val_noised = noise_data(val_data, noise_type=noise_type, gaussian_mean=0, gaussian_std=eval_gaussian_std)
        val_normalized_noised = ((val_noised - val_dataset.norm_mean) / val_dataset.norm_std).astype(np.float32)
        val_normalized = ((val_data - val_dataset.norm_mean) / val_dataset.norm_std).astype(np.float32)

    # val_noised = (val_noised * 255).astype(np.uint8)
    # val_noised = noise_data(val_dataset.X / 255, noise_type=noise_type, gaussian_mean=0,
    #                         gaussian_std=eval_gaussian_std)
    # val_dataset_noised = val_dataset.duplicate()
    # val_dataset_noised.X = (val_noised * 255).astype(np.uint8)
    # val_normalized_noised = (val_noised - val_dataset.norm_mean) / val_dataset.norm_std

    # val_loader = val_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    print(f'Z val generation...')
    # val_inZ = []
    # elabels = []
    val_inZ = []
    with torch.no_grad():
        n_iter = math.ceil(val_normalized_noised.shape[0] / batch_size)
        for i in range(n_iter):
            inp = torch.from_numpy(val_normalized_noised[i * batch_size:(i + 1) * batch_size])
            inp = train_dataset.format_data(inp, device)
            labels = torch.from_numpy(elabels[i * batch_size:(i + 1) * batch_size]).to(device)
            log_p, distloss, logdet, out = model(inp, labels)
            val_inZ.append(out.detach().cpu().numpy().reshape(inp.shape[0], -1))
    val_inZ = np.concatenate(val_inZ, axis=0)
    #     for j, data in enumerate(val_loader):
    #         inp, labels = data
    #         inp = train_dataset.format_data(inp, device)
    #         labels = labels.to(device)
    #         log_p, distloss, logdet, out = model(inp, labels)
    #         val_inZ.append(out.detach().cpu().numpy())
    #         elabels.append(labels.detach().cpu().numpy())
    # val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(val_dataset), -1)
    # elabels = np.concatenate(elabels, axis=0)

    # val_normalized = (val_dataset.X / 255 - val_dataset.norm_mean) / val_dataset.norm_std
    grid_im = [[] for _ in range(len(gaussian_params) * how_much_by_label)]
    vis_index = 0
    # add in grid ori and noisy
    for i, gaussian_param in enumerate(gaussian_params):
        label = gaussian_param[-1]
        if label in val_dataset.true_labels:
            # grid_im[i].append(np.expand_dims(val_normalized[np.where(elabels == label)][vis_index], axis=0))
            i_grid = i * how_much_by_label
            for n in range(how_much_by_label):
                grid_im[i_grid + n].append(
                    np.expand_dims(val_normalized_noised[np.where(elabels == label)][vis_index + n], axis=0))

    for n in range(len(how_much)):
        n_dim_projection = how_much[n]
        print(f'Projections on {n_dim_projection} processing...')
        # ZPCA
        if proj_type == 'zpca':
            # ZPCA
            zpca = PCA(n_components=n_dim_projection)
            zpca.fit(Z)
            pca_projection = zpca.transform(val_inZ)
            proj = zpca.inverse_transform(pca_projection)
            ordered_elabels = elabels
        else:
            projs = []
            ordered_val = []
            ordered_elabels = []
            for i, gaussian_param in enumerate(gaussian_params):
                gmean = model.means[i].detach().cpu().numpy()
                gp = gaussian_param[1:-1]
                label = gaussian_param[-1]
                if label in elabels:
                    indexes = np.where(elabels == label)[0]
                    val_Z_lab = val_inZ[indexes]
                    val_normalized_lab = val_normalized[indexes]

                    ordered_elabels.append(np.array([label for _ in range(indexes.shape[0])]))

                    if proj_type == 'zpca_l':
                        # Z-PCA
                        Z_lab = Z[np.where(tlabels == label)]
                        pca = PCA(n_components=n_dim_projection)
                        pca.fit(Z_lab)
                        pca_projection = pca.transform(val_Z_lab)
                        proj = pca.inverse_transform(pca_projection)
                    else:
                        proj = project_inZ(val_Z_lab, params=(gmean, gp), how_much=n_dim_projection)
                    projs.append(proj)
                    ordered_val.append(val_normalized_lab)
            ordered_elabels = np.concatenate(ordered_elabels, axis=0)
            proj = np.concatenate(projs, axis=0)

        nb_batch = math.ceil(proj.shape[0] / batch_size)
        all_im = []
        proj = torch.from_numpy(proj)
        for j in range(nb_batch):
            size = proj[j * batch_size:(j + 1) * batch_size].shape[0]
            z_b = proj[j * batch_size:(j + 1) * batch_size].reshape(size, *z_shape).float().to(device)

            images = model.reverse(z_b).cpu().data.numpy()
            all_im.append(images)
        all_im = np.concatenate(all_im, axis=0)

        for i, gaussian_param in enumerate(gaussian_params):
            label = gaussian_param[-1]
            if label in ordered_elabels:
                i_grid = i * how_much_by_label
                for n in range(how_much_by_label):
                    ind = np.where(ordered_elabels == label)[0]
                    grid_im[i_grid + n].append(np.expand_dims(all_im[ind][vis_index + n], axis=0))

    grid_im = [item for sublist in grid_im for item in sublist]
    grid_im = np.concatenate(grid_im, axis=0)
    grid_im = val_dataset.rescale(grid_im)

    # used_method = ['ori', 'noisy']
    used_method = ['noisy']
    for n in range(len(how_much)):
        used_method += [proj_type + str(how_much[n])]
    methods = used_method * len(gaussian_params) * how_much_by_label
    labels = [[str(g[-1]) + f'_{n}'] * len(used_method) for g in gaussian_params for n in range(how_much_by_label)]
    labels = [item for sublist in labels for item in sublist]
    save_every_pic(f'{save_dir}/compressions/{noise_type}/every_pics/{proj_type}', grid_im, methods, labels,
                   add_str=proj_type, rgb=val_dataset.n_channel > 1)

    nrow = math.floor(grid_im.shape[0] / (np.unique(val_dataset.true_labels).shape[0] * how_much_by_label))
    utils.save_image(
        torch.from_numpy(grid_im),
        f"{save_dir}/compressions/{noise_type}/compression_eval_{proj_type}.png",
        normalize=True,
        nrow=nrow,
        range=(0, 255),
    )

    return grid_im


def evaluate_p_value(predictions, by_pairs=False):
    preds = [v for _, v in predictions.items()]
    if len(preds) > 1:
        if by_pairs:
            i_method = len(preds) - 1
            res = {}
            for i, (k, v) in enumerate(predictions.items()):
                if i != i_method:
                    H, p = kruskal(preds[i_method], preds[i])
                    res[k] = (H, p)
            return res
        else:
            H, p = kruskal(*preds)
            return H, p
    else:
        return None


def evaluate_classification(model, train_dataset, val_dataset, save_dir, device, fithyperparam=True, batch_size=10):
    if isinstance(train_dataset, GraphDataset):
        train_dataset.permute_graphs_in_dataset()
        val_dataset.permute_graphs_in_dataset()

    zlinsvc = None

    # Compute results with our approach if not None
    if model is not None:
        model.eval()
        start = time.time()
        Z = []
        tlabels = []
        if isinstance(train_dataset, GraphDataset):
            n_permutation = 5
            for n_perm in range(n_permutation):
                train_dataset.permute_graphs_in_dataset()

                loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

                with torch.no_grad():
                    for j, data in enumerate(loader):
                        inp, labels = data
                        inp = train_dataset.format_data(inp, device)
                        labels = labels.to(device)
                        log_p, distloss, logdet, out = model(inp, labels)
                        Z.append(out.detach().cpu().numpy())
                        tlabels.append(labels.detach().cpu().numpy())
        else:
            loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

            with torch.no_grad():
                for j, data in enumerate(loader):
                    inp, labels = data
                    inp = train_dataset.format_data(inp, device)
                    labels = labels.to(device)
                    log_p, distloss, logdet, out = model(inp, labels)
                    Z.append(out.detach().cpu().numpy())
                    tlabels.append(labels.detach().cpu().numpy())
        tlabels = np.concatenate(tlabels, axis=0)
        Z = np.concatenate(Z, axis=0).reshape(tlabels.shape[0], -1)
        end = time.time()
        print(f"time to get Z_train from X_train: {str(end - start)}, batch size: {batch_size}")

        # Learn SVC
        start = time.time()
        kernel_name = 'zlinear'
        if fithyperparam:
            param_gridlin = [
                {'SVC__kernel': ['linear'], 'SVC__C': np.concatenate((np.logspace(-5, 3, 10), np.array([1])))}]
            # param_gridlin = [{'SVC__kernel': ['linear'], 'SVC__C': np.array([1])}]
            model_type = ('SVC', SVC(max_iter=10000))
            # model_type = ('SVC', SVC())
            scaler = True
            zlinsvc = learn_or_load_modelhyperparams(Z, tlabels, kernel_name, param_gridlin, save_dir,
                                                     model_type=model_type, scaler=scaler, save=False)
        else:
            zlinsvc = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))
            zlinsvc.fit(Z, tlabels)
            print(f'Fitting done.')
        print(zlinsvc)
        end = time.time()
        print(f"time to fit linear svc in Z : {str(end - start)}")

    # KERNELS FIT
    X_train = train_dataset.get_flattened_X()  # TODO : add the permutations ?
    labels_train = train_dataset.true_labels

    start = time.time()
    kernel_name = 'linear'
    if fithyperparam:
        param_gridlin = [
            {'SVC__kernel': [kernel_name], 'SVC__C': np.concatenate((np.logspace(-5, 3, 10), np.array([1])))}]
        linsvc = learn_or_load_modelhyperparams(X_train, labels_train, kernel_name, param_gridlin, save_dir,
                                                model_type=('SVC', SVC()), scaler=False)
    else:
        linsvc = make_pipeline(StandardScaler(), SVC(kernel=kernel_name, C=1.0))
        linsvc.fit(X_train, labels_train)
        print(f'Fitting done.')
    end = time.time()
    print(f"time to fit linear svc : {str(end - start)}")

    # krr_types = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    ksvc_types = ['rbf', 'poly', 'sigmoid']
    # ksvc_types = ['poly']
    ksvc_params = [
        {'SVC__kernel': ['rbf'], 'SVC__gamma': np.logspace(-5, 3, 10),
         'SVC__C': np.concatenate((np.logspace(-5, 3, 10), np.array([1])))},
        # {'SVC__kernel': ['poly'], 'SVC__gamma': np.logspace(-5, 3, 5),
        #  'SVC__degree': np.linspace(1, 4, 4).astype(np.int),
        #  'SVC__C': np.concatenate((np.logspace(-5, 3, 10), np.array([1])))},
        {'SVC__kernel': ['poly'], 'SVC__gamma': np.logspace(-5, 1, 5),
         'SVC__degree': np.linspace(1, 2, 2).astype(np.int),
         'SVC__C': np.concatenate((np.logspace(-5, 1, 10), np.array([1])))},
        {'SVC__kernel': ['sigmoid'], 'SVC__C': np.concatenate((np.logspace(-5, 3, 10), np.array([1])))}
    ]

    ksvcs = [None] * len(ksvc_types)
    for i, ksvc_type in enumerate(ksvc_types):
        start = time.time()
        if fithyperparam:
            ksvcs[i] = learn_or_load_modelhyperparams(X_train, labels_train, ksvc_type, [ksvc_params[i]], save_dir,
                                                      model_type=('SVC', SVC()), scaler=True)
        else:
            ksvcs[i] = make_pipeline(StandardScaler(), SVC(kernel=ksvc_type, C=1.0))
            ksvcs[i].fit(X_train, labels_train)
            print(f'Fitting done.')
        end = time.time()
        print(f"time to fit {ksvc_type} svc : {str(end - start)}")

    # GRAPH KERNELS FIT
    if isinstance(train_dataset, GraphDataset):
        def compute_kernel(name, dataset, edge_to_node, normalize, wl_height=10, attributed_node=False):
            if name == 'wl':
                K = compute_wl_kernel(dataset, wl_height=wl_height, edge_to_node=edge_to_node,
                                      normalize=normalize, attributed_node=attributed_node)
            elif name == 'prop':
                K = compute_propagation_kernel(dataset, normalize=normalize, edge_to_node=edge_to_node,
                                               attributed_node=attributed_node)
            elif name == 'sp':
                K = compute_sp_kernel(dataset, normalize=normalize, edge_to_node=edge_to_node,
                                      attributed_node=attributed_node)
            elif name == 'mslap':
                K = compute_mslap_kernel(dataset, normalize=normalize, edge_to_node=edge_to_node,
                                         attributed_node=attributed_node)
            elif name == 'hadcode':
                K = compute_hadcode_kernel(dataset, normalize=normalize, edge_to_node=edge_to_node,
                                           attributed_node=attributed_node)
            else:
                assert False, f'unknown graph kernel: {graph_kernel}'
            return K

        wl_height = 5
        normalize = False
        if val_dataset.is_attributed_node_dataset():
            # graph_kernel_names = ['mslap']
            # graph_kernel_names = ['prop']
            graph_kernel_names = []
            attributed_node = True
            edge_to_node = False
        else:
            graph_kernel_names = ['wl', 'sp', 'hadcode']
            attributed_node = False
            edge_to_node = True
        graph_kernels = []
        graph_svc_params = []
        for graph_kernel in graph_kernel_names:
            K = compute_kernel(graph_kernel, train_dataset, edge_to_node=edge_to_node, normalize=normalize,
                               wl_height=wl_height, attributed_node=attributed_node)
            graph_kernels.append(('precomputed', K, graph_kernel))
            graph_svc_params.append(
                {'SVC__kernel': ['precomputed'], 'SVC__C': np.concatenate((np.logspace(-5, 3, 10), np.array([1])))})

        graph_ksvcs = [None] * len(graph_kernels)
        for i, (krr_type, K, name) in enumerate(graph_kernels):
            start = time.time()
            if fithyperparam:
                graph_ksvcs[i] = learn_or_load_modelhyperparams(K, labels_train, name, [graph_svc_params[i]],
                                                                save_dir,
                                                                model_type=('SVC', SVC()), scaler=False)
            else:
                graph_ksvcs[i] = make_pipeline(StandardScaler(), SVC(kernel=name, C=1.0))
                graph_ksvcs[i].fit(K, labels_train)
                print(f'Fitting done.')
            end = time.time()
            print(f"time to fit {krr_type} ridge : {str(end - start)}")

    if isinstance(train_dataset, GraphDataset):
        n_permutation = 20
        our_preds = []
        our_scores = []
        # our_pred_scores = []
        our_scores_train = []
        svc_preds = []
        svc_scores = []
        ksvc_preds = []
        ksvc_scores = []
        for ksvc in ksvcs:
            ksvc_scores.append([])
            ksvc_preds.append([])
        ksvc_graph_preds = []
        ksvc_graph_scores = []
        for graph_ksvc in graph_ksvcs:
            ksvc_graph_scores.append([])
            ksvc_graph_preds.append([])

        for n_perm in range(n_permutation):
            val_dataset.permute_graphs_in_dataset()
            # OUR APPROACH EVALUATION
            if model is not None:
                # batch_size = 200 if 200 < len(val_dataset) else int(len(val_dataset) / 2)
                val_loader = val_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

                start = time.time()
                val_inZ = []
                elabels = []
                val_predict = []
                with torch.no_grad():
                    for j, data in enumerate(val_loader):
                        inp, labels = data
                        inp = val_dataset.format_data(inp, device)
                        labels = labels.to(device)
                        log_p, distloss, logdet, out = model(inp, labels)
                        # probs = predict_by_log_p_with_gaussian_params(out, model.means, model.gaussian_params)
                        # val_predict.append(probs.detach().cpu().numpy())
                        val_inZ.append(out.detach().cpu().numpy())
                        elabels.append(labels.detach().cpu().numpy())
                val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(val_dataset), -1)
                elabels = np.concatenate(elabels, axis=0)
                # val_predict = np.concatenate(val_predict, axis=0)
                end = time.time()
                print(f"time to get Z_val from X_val : {str(end - start)}")

                start = time.time()
                zsvc_pred = zlinsvc.predict(val_inZ)
                zsvc_score = classification_score(zsvc_pred, elabels)
                # val_pred_scores = classification_score(val_predict, elabels)
                # zsvc_score = zlinsvc.score(val_inZ, elabels)
                our_preds.append(zsvc_pred)
                our_scores.append(zsvc_score)
                # our_pred_scores.append(val_pred_scores)
                end = time.time()
                print(f"time to predict with zlinSVC : {str(end - start)}")

                # See on train
                start = time.time()
                zsvc_score = zlinsvc.score(Z, tlabels)
                our_scores_train.append(zsvc_score)
                end = time.time()
                print(f"time to predict with zlinSVC (on train) : {str(end - start)}")

            # KERNELS EVALUATION
            X_val = val_dataset.get_flattened_X()
            labels_val = val_dataset.true_labels

            start = time.time()
            svc_pred = linsvc.predict(X_val)
            svc_score = classification_score(svc_pred, labels_val)
            svc_preds.append(svc_pred)
            svc_scores.append(svc_score)
            end = time.time()
            print(f"time to predict with xlinSVC : {str(end - start)}")

            start = time.time()
            for i, ksvc in enumerate(ksvcs):
                ksvc_pred = ksvc.predict(X_val)
                ksvc_score = classification_score(ksvc_pred, labels_val)
                # ksvc_score = ksvc.score(X_val, labels_val)
                ksvc_preds[i].append(ksvc_pred)
                ksvc_scores[i].append(ksvc_score)
            end = time.time()
            print(f"time to predict with {len(ksvcs)} kernelSVC : {str(end - start)}")

            start = time.time()
            # GRAPH KERNELS EVALUATION
            for i, graph_ksvc in enumerate(graph_ksvcs):
                K_val = compute_kernel(graph_kernels[i][2], (val_dataset, train_dataset), edge_to_node=edge_to_node,
                                       normalize=normalize, wl_height=wl_height, attributed_node=attributed_node)
                graph_ksvc_pred = graph_ksvc.predict(K_val)
                graph_ksvc_score = classification_score(graph_ksvc_pred, labels_val)
                # graph_ksvc_score = graph_ksvc.score(K_val, labels_val)
                ksvc_graph_preds[i].append(graph_ksvc_pred)
                ksvc_graph_scores[i].append(graph_ksvc_score)
            end = time.time()
            print(f"time to predict with {len(graph_ksvcs)} graphkernelSVC : {str(end - start)}")

        # PRINT RESULTS
        lines = []
        print('Predictions scores :')

        # P-value calculation
        # predictions = {'linear': np.concatenate(svc_preds)}
        # for ktype, kpred in zip(ksvc_types, ksvc_preds):
        #     predictions[ktype] = np.concatenate(kpred)
        # for ktype, kpred in zip(graph_kernels, ksvc_graph_preds):
        #     predictions[ktype] = np.concatenate(kpred)
        # if model is not None:
        #     predictions['zlinear'] = np.concatenate(our_preds)
        # res_pvalue = evaluate_p_value(predictions)
        # if res_pvalue is not None:
        #     H, p = res_pvalue
        #     score_str = 'Kruskal-Wallis H-test, H: ' + str(H) + ', p-value: ' + str(p)
        #     print(score_str)
        #     lines += [score_str, '\n']

        # for i, pred1 in enumerate(preds):
        #     for j, pred2 in enumerate(preds):
        #         if i != j:
        #             U, p2 = mannwhitneyu(pred1, pred2, alternative='two-sided')
        #             print(f'({i},{j}):' + str(p2))

        svc_mean_score = np.mean(svc_scores)
        svc_std_score = np.std(svc_scores)
        score_str = f'SVC R2: {svc_scores} \n' \
                    f'Mean Scores: {svc_mean_score} \n' \
                    f'Std Scores: {svc_std_score}'
        print(score_str)
        lines += [score_str, '\n']

        for j, ksvc_type in enumerate(ksvc_types):
            scores = []
            for ksvc_score in ksvc_scores[j]:
                scores.append(ksvc_score)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            score_str = f'KernelSVC ({ksvc_type}): {scores} \n' \
                        f'Mean Scores: {mean_score} \n' \
                        f'Std Scores: {std_score}'
            print(score_str)
            lines += [score_str, '\n']

        for j, graph_ksvc in enumerate(graph_ksvcs):
            scores = []
            for graph_ksvc_score in ksvc_graph_scores[j]:
                scores.append(graph_ksvc_score)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            score_str = f'GraphKernelSVC ({graph_kernels[j][2]}): {scores} \n' \
                        f'Mean Scores: {mean_score} \n' \
                        f'Std Scores: {std_score}'
            print(score_str)
            lines += [score_str, '\n']

        mean_score = np.mean(our_scores)
        # mean_pred_score = np.mean(our_pred_scores)
        std_score = np.std(our_scores)
        score_str = f'Our approach {our_scores} \n' \
                    f'Mean Scores: {mean_score} \n' \
                    f'Std Scores: {std_score}'
        print(score_str)
        lines += [score_str, '\n']
        mean_score = np.mean(our_scores_train)
        std_score = np.std(our_scores_train)
        score_str = f'(On train) Our approach: {our_scores_train} \n' \
                    f'Mean Scores: {mean_score} \n' \
                    f'Std Scores: {std_score}'
        print(score_str)
        lines += [score_str, '\n']

    else:
        # OUR APPROACH EVALUATION
        if model is not None:
            val_loader = val_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

            start = time.time()
            val_inZ = []
            elabels = []
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    inp, labels = data
                    inp = val_dataset.format_data(inp, device)
                    labels = labels.to(device)
                    log_p, distloss, logdet, out = model(inp, labels)
                    val_inZ.append(out.detach().cpu().numpy())
                    elabels.append(labels.detach().cpu().numpy())
            val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(val_dataset), -1)
            elabels = np.concatenate(elabels, axis=0)
            end = time.time()
            print(f"time to get Z_val from X_val : {str(end - start)}")

            zsvc_pred = zlinsvc.predict(val_inZ)
            zsvc_score = classification_score(zsvc_pred, elabels)
            # zsvc_score = zlinsvc.score(val_inZ, elabels)

            print(f'Our approach : {zsvc_score}')

            t_zsvc_pred = zlinsvc.predict(Z)
            t_zsvc_score = classification_score(t_zsvc_pred, tlabels)
            # t_zsvc_score = zlinsvc.score(Z, tlabels)
            print(f'(On Train) Our approach : {t_zsvc_score}')

            # Misclassified data
            predictions = zlinsvc.predict(val_inZ)
            misclassif_i = np.where((predictions == elabels) == False)
            if misclassif_i[0].shape[0] > 0:
                z_shape = model.calc_last_z_shape(val_dataset.in_size)
                z_sample = torch.from_numpy(
                    val_inZ[misclassif_i].reshape(misclassif_i[0].shape[0], *z_shape)).float().to(
                    device)
                with torch.no_grad():
                    images = model.reverse(z_sample).cpu().data
                images = val_dataset.rescale(images)
                print('Misclassification:')
                print('Real labels :' + str(elabels[misclassif_i]))
                print('Predicted labels :' + str(predictions[misclassif_i]))

            if train_dataset.dataset_name == 'mnist':
                nrow = math.floor(math.sqrt(images.shape[0]))
                utils.save_image(
                    images,
                    f"{save_dir}/misclassif.png",
                    normalize=True,
                    nrow=nrow,
                    range=(0, 255),
                )

                # write in images pred and true labels
                ndarr = images.numpy()
                res = []
                margin = 10
                shape = (ndarr[0].shape[0], ndarr[0].shape[1] + margin, ndarr[0].shape[2] + 2 * margin)
                for j, im_arr in enumerate(ndarr):
                    im = np.zeros(shape)
                    im[:, margin:, margin:-margin] = im_arr[:, :, :]
                    im = im.squeeze()
                    img = Image.fromarray(im)
                    draw = ImageDraw.Draw(img)
                    draw.text((0, 0), f'P:{predictions[misclassif_i[0][j]]} R:{elabels[misclassif_i[0][j]]}', 255)
                    res.append(np.expand_dims(np.array(img).reshape(shape), axis=0))
                res = torch.from_numpy(np.concatenate(res, axis=0)).to(torch.float32)
                utils.save_image(
                    res,
                    f"{save_dir}/misclassif_PR.png",
                    normalize=True,
                    nrow=nrow,
                    range=(0, 255),
                )

        # KERNELS EVALUATION
        X_val = val_dataset.get_flattened_X()
        labels_val = val_dataset.true_labels

        ksvc_preds = []
        ksvc_scores = []
        for ksvc in ksvcs:
            ksvc_scores.append([])

        svc_pred = linsvc.predict(X_val)
        svc_score = classification_score(svc_pred, labels_val)

        for i, ksvc in enumerate(ksvcs):
            ksvc_pred = ksvc.predict(X_val)
            ksvc_score = classification_score(ksvc_pred, labels_val)
            # ksvc_score = ksvc.score(X_val, labels_val)
            ksvc_scores[i].append(ksvc_score)
            ksvc_preds.append(ksvc_pred)

        lines = []
        print('Predictions scores :')

        # P-value calculation
        predictions = {'linear': svc_pred}
        for ktype, kpred in zip(ksvc_types, ksvc_preds):
            predictions[ktype] = kpred
        if model is not None:
            predictions['zlinear'] = zsvc_pred
        res_pvalue = evaluate_p_value(predictions)
        if res_pvalue is not None:
            H, p = res_pvalue
            score_str = 'Kruskal-Wallis H-test, H: ' + str(H) + ', p-value: ' + str(p)
            print(score_str)
            lines += [score_str, '\n']

        # score_str = f'SVC Linear: {svc_score}'
        # print(score_str)
        # lines += [score_str, '\n']
        for j, krr_type in enumerate(ksvc_types):
            score_str = f'KernelRidge ({krr_type}):{np.mean(ksvc_scores[j])}'
            print(score_str)
            lines += [score_str, '\n']

        if model is not None:
            score_str = f'Our approach: {zsvc_score}'
            print(score_str)
            lines += [score_str, '\n']
            score_str = f'(On train) Our approach: {t_zsvc_score}'
            print(score_str)
            lines += [score_str, '\n']

    with open(f"{save_dir}/eval_res.txt", 'w') as f:
        f.writelines(lines)

    # np.save(f"{save_dir}/predictions.npy", predictions)

    return zlinsvc


def generate_meanclasses(model, val_dataset, device, save_dir, debug=False, batch_size=20):
    model.eval()
    create_folder(f'{save_dir}/test_generation')

    loader = val_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    Z = []
    tlabels = []
    with torch.no_grad():
        for j, data in enumerate(loader):
            inp = data[0].float().to(device)
            labels = data[1].to(device)
            log_p, distloss, logdet, out = model(inp, labels)
            Z.append(out.detach().cpu().numpy())
            tlabels.append(labels.detach().cpu().numpy())
    Z = np.concatenate(Z, axis=0)
    tlabels = np.concatenate(tlabels, axis=0)

    Z_means = []
    for label in np.unique(tlabels):
        Z_means.append(np.expand_dims(np.mean(Z[tlabels == label], axis=0), axis=0))
    Z_means = np.concatenate(Z_means, axis=0)

    z_sample = torch.from_numpy(Z_means).float().to(device)
    with torch.no_grad():
        if debug:
            create_folder(f"{save_dir}/test_generation/mean_flows_gen")
            images = model.reverse_debug(z_sample, val_dataset,
                                         save_dir=f"{save_dir}/test_generation/mean_flows_gen").cpu().data
        else:
            images = model.reverse(z_sample).cpu().data
    images = val_dataset.rescale(images)

    utils.save_image(
        images,
        f"{save_dir}/test_generation/meanclasses.png",
        normalize=True,
        nrow=images.shape[0],
        range=(0, 255),
    )

    return images


def evaluate_distances(model, train_dataset, val_dataset, gaussian_params, z_shape, how_much, kpca_types, device,
                       save_dir='./save', proj_type='gp', noise_type='gaussian', eval_gaussian_std=.1, batch_size=20):
    create_folder(f'{save_dir}/projections')

    train_dataset.ori_X = train_dataset.X
    val_dataset.ori_X = val_dataset.X
    train_dataset.ori_true_labels = train_dataset.true_labels
    val_dataset.ori_true_labels = val_dataset.true_labels
    train_dataset.idx = np.array([i for i in range(train_dataset.X.shape[0])])
    val_dataset.idx = np.array([i for i in range(val_dataset.X.shape[0])])

    train_noised = noise_data(train_dataset.X, noise_type=noise_type, gaussian_mean=0,
                              gaussian_std=eval_gaussian_std, clip=False)
    train_dataset_noised = train_dataset.duplicate()
    train_dataset_noised.X = train_noised

    val_noised = noise_data(val_dataset.X, noise_type=noise_type, gaussian_mean=0,
                            gaussian_std=eval_gaussian_std, clip=False)
    val_dataset_noised = val_dataset.duplicate()
    val_dataset_noised.X = val_noised

    loader = train_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    if proj_type in ['zpca', 'zpca_l']:
        train_noised_Z = []
        tlabels = []
        with torch.no_grad():
            for j, data in enumerate(loader):
                inp = data[0].float().to(device)
                labels = data[1].to(torch.int64).to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                train_noised_Z.append(out.detach().cpu().numpy())
                tlabels.append(labels.detach().cpu().numpy())
        train_noised_Z = np.concatenate(train_noised_Z, axis=0).reshape(len(train_dataset), -1)
        tlabels = np.concatenate(tlabels, axis=0)

    val_loader = val_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    val_noised_inZ = []
    elabels = []
    with torch.no_grad():
        for j, data in enumerate(val_loader):
            inp = data[0].float().to(device)
            labels = data[1].to(torch.int64).to(device)
            log_p, distloss, logdet, out = model(inp, labels)
            val_noised_inZ.append(out.detach().cpu().numpy())
            elabels.append(labels.detach().cpu().numpy())
    val_noised_inZ = np.concatenate(val_noised_inZ, axis=0).reshape(len(val_dataset), -1)
    elabels = np.concatenate(elabels, axis=0)

    # ZPCA
    if proj_type == 'zpca':
        # KPCA
        kpca_projs = []
        for j, kpca_type in enumerate(kpca_types):
            print(f'{kpca_type}-PCA processing...')
            kpca = KernelPCA(n_components=how_much, kernel=kpca_type, fit_inverse_transform=True)
            kpca.fit(train_noised.reshape(train_noised.shape[0], -1))

            kpca_projection = kpca.transform(val_noised.reshape(val_noised.shape[0], -1))
            kpca_reconstruct = kpca.inverse_transform(kpca_projection)
            kpca_projs.append(kpca_reconstruct)

        # ZPCA
        zpca = PCA(n_components=how_much)
        zpca.fit(train_noised_Z)
        pca_projection = zpca.transform(val_noised_inZ)
        proj = zpca.inverse_transform(pca_projection)
        ordered_val = val_dataset.X
        ordered_elabels = elabels
    else:
        kpca_projs = []
        for type in kpca_types:
            kpca_projs.append([])
        projs = []
        ordered_val = []
        ordered_elabels = []
        for i, gaussian_param in enumerate(gaussian_params):
            gmean = model.means[i].detach().cpu().numpy()
            gp = gaussian_param[1:-1]
            label = gaussian_param[-1]

            train_noised_lab = train_noised[np.where(train_dataset_noised.true_labels == label)]
            indexes = np.where(elabels == label)[0]
            val_noised_inZ_lab = val_noised_inZ[indexes]
            val_lab = val_dataset.X[indexes]
            val_noised_lab = val_noised[np.where(val_dataset_noised.true_labels == label)]

            ordered_elabels.append(np.array([label for _ in range(indexes.shape[0])]))

            # K-PCA
            for j, kpca_type in enumerate(kpca_types):
                print(f'{kpca_type}-PCA (label {label}) processing...')
                kpca = KernelPCA(n_components=how_much, kernel=kpca_type, fit_inverse_transform=True)
                kpca.fit(train_noised_lab.reshape(train_noised_lab.shape[0], -1))

                kpca_projection = kpca.transform(
                    val_noised_lab.reshape(val_noised_lab.shape[0], -1))
                kpca_reconstruct = kpca.inverse_transform(kpca_projection)
                kpca_projs[j].append(kpca_reconstruct)

            if proj_type == 'zpca_l':
                # Z-PCA
                train_noised_Z_lab = train_noised_Z[np.where(tlabels == label)]
                zpca = PCA(n_components=how_much)
                zpca.fit(train_noised_Z_lab)
                pca_projection = zpca.transform(val_noised_inZ_lab)
                proj = zpca.inverse_transform(pca_projection)
            else:
                proj = project_inZ(val_noised_inZ_lab, params=(gmean, gp), how_much=how_much)
            projs.append(proj)
            ordered_val.append(val_lab)
        ordered_elabels = np.concatenate(ordered_elabels, axis=0)
        ordered_val = np.concatenate(ordered_val, axis=0)
        for k, kpca_proj in enumerate(kpca_projs):
            kpca_projs[k] = np.concatenate(kpca_proj, axis=0)
        proj = np.concatenate(projs, axis=0)

    distances_results = {}
    # KPCA results
    for j, kpca_type in enumerate(kpca_types):
        mean_dist = np.mean(np.sum(np.abs(kpca_projs[j] - ordered_val.reshape(ordered_val.shape[0], -1)), axis=1))
        print(f'{kpca_type}-PCA: {mean_dist}')
        distances_results[f'{kpca_type}-PCA'] = mean_dist

    # Our approach results
    nb_batch = math.ceil(proj.shape[0] / batch_size)
    all_res = []
    proj = torch.from_numpy(proj)
    for j in range(nb_batch):
        size = proj[j * batch_size:(j + 1) * batch_size].shape[0]
        z_b = proj[j * batch_size:(j + 1) * batch_size].reshape(size, *z_shape).float().to(device)

        res = model.reverse(z_b)
        all_res.append(res.detach().cpu().numpy())
    all_res = np.concatenate(all_res, axis=0)

    our_dist = np.sum(np.abs(all_res.reshape(all_res.shape[0], -1) - ordered_val.reshape(ordered_val.shape[0], -1)),
                      axis=1)
    mean_dist = np.mean(our_dist)
    print(f'Our approach ({proj_type}) : {mean_dist}')
    distances_results[f'Our-{proj_type}'] = mean_dist

    base_dist = np.mean(np.sum(np.abs(val_noised - ordered_val.reshape(ordered_val.shape[0], -1)), axis=1))
    distances_results['NoiseDist'] = base_dist
    print(f'Noise base distance : {base_dist}')
    # from get_best_projection import save_fig
    # save_fig(ordered_val, all_res, ordered_elabels, label_max=1, size=5,
    #          save_path=f'{save_dir}/projection_zpca_model')

    return distances_results


def evaluate_regression(model, train_dataset, val_dataset, save_dir, device, fithyperparam=True, save_res=True,
                        batch_size=200):
    if isinstance(train_dataset, GraphDataset):
        train_dataset.permute_graphs_in_dataset()
        val_dataset.permute_graphs_in_dataset()

    zlinridge = None
    # Compute results with our approach if not None
    if model is not None:
        loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

        start = time.time()
        Z = []
        tlabels = []
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(loader):
                inp, labels = data
                inp = train_dataset.format_data(inp, device)
                labels = labels.to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                Z.append(out.detach().cpu().numpy())
                tlabels.append(labels.detach().cpu().numpy())
        Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)
        tlabels = np.concatenate(tlabels, axis=0)
        end = time.time()
        print(f"time to get Z_train from X_train: {str(end - start)}, batch size: {batch_size}")

        # Learn Ridge
        start = time.time()
        kernel_name = 'zlinear'
        if fithyperparam:
            # param_gridlin = [{'Ridge__kernel': [kernel_name], 'Ridge__alpha': np.linspace(0, 10, 11)}]
            # linridge = learn_or_load_modelhyperparams(X_train, labels_train, kernel_name, param_gridlin, save_dir,
            #                                           model_type=('Ridge', KernelRidge()), scaler=True)
            # param_gridlin = [{'Ridge__kernel': [kernel_name], 'Ridge__alpha': np.logspace(-5, 2, 11)}]
            param_gridlin = [{'Ridge__alpha': np.concatenate((np.logspace(-5, 2, 11), np.array([1])))}]
            zlinridge = learn_or_load_modelhyperparams(Z, tlabels, kernel_name, param_gridlin, save_dir,
                                                       model_type=('Ridge', Ridge()), scaler=False, save=False)
        else:
            # linridge = make_pipeline(StandardScaler(), KernelRidge(kernel=kernel_name))
            zlinridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            zlinridge.fit(Z, tlabels)
            print(f'Fitting done.')
        # zlinridge = make_pipeline(StandardScaler(), KernelRidge(kernel='linear', alpha=0.1))
        # zlinridge = Ridge(alpha=0.1)
        # zlinridge.fit(Z, tlabels)
        print(zlinridge)
        end = time.time()
        print(f"time to fit linear ridge in Z : {str(end - start)}")

    # KERNELS FIT
    X_train = train_dataset.get_flattened_X()
    # X_train = train_dataset.X.reshape(train_dataset.X.shape[0], -1)
    labels_train = train_dataset.true_labels

    start = time.time()
    kernel_name = 'linear'
    if fithyperparam:
        # param_gridlin = [{'Ridge__kernel': [kernel_name], 'Ridge__alpha': np.linspace(0, 10, 11)}]
        # linridge = learn_or_load_modelhyperparams(X_train, labels_train, kernel_name, param_gridlin, save_dir,
        #                                           model_type=('Ridge', KernelRidge()), scaler=True)
        # param_gridlin = [{'Ridge__kernel': [kernel_name], 'Ridge__alpha': np.logspace(-5, 2, 11)}]
        param_gridlin = [{'Ridge__alpha': np.logspace(-5, 2, 11)}]
        linridge = learn_or_load_modelhyperparams(X_train, labels_train, kernel_name, param_gridlin, save_dir,
                                                  model_type=('Ridge', Ridge()), scaler=False)
    else:
        # linridge = make_pipeline(StandardScaler(), KernelRidge(kernel=kernel_name))
        linridge = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
        linridge.fit(X_train, labels_train)
        print(f'Fitting done.')
    end = time.time()
    print(f"time to fit linear ridge : {str(end - start)}")

    # krr_types = ['rbf', 'poly', 'sigmoid']
    # krr_types = ['rbf']
    krr_types = ['poly', 'sigmoid']
    # krr_types = []
    krr_params = [
        # {'Ridge__kernel': ['rbf'], 'Ridge__gamma': np.logspace(-5, 3, 5), 'Ridge__alpha': np.logspace(-5, 2, 11)},
        {'Ridge__kernel': ['poly'], 'Ridge__gamma': np.logspace(-5, 3, 5),
         'Ridge__degree': np.linspace(1, 4, 4).astype(np.int),
         'Ridge__alpha': np.logspace(-5, 2, 11)},
        {'Ridge__kernel': ['sigmoid'], 'Ridge__gamma': np.logspace(-5, 3, 5), 'Ridge__alpha': np.logspace(-5, 2, 11)}
    ]
    # krr_params = [
    #     {'Ridge__kernel': ['rbf'], 'Ridge__alpha': np.linspace(0, 10, 11)},
    #     {'Ridge__kernel': ['poly'], 'Ridge__degree': np.linspace(1, 4, 4).astype(np.int),
    #      'Ridge__alpha': np.linspace(0, 10, 11)},
    #     {'Ridge__kernel': ['sigmoid'], 'Ridge__alpha': np.linspace(0, 10, 11)}
    # ]
    krrs = [None] * len(krr_types)
    for i, krr_type in enumerate(krr_types):
        start = time.time()
        if fithyperparam:
            krrs[i] = learn_or_load_modelhyperparams(X_train, labels_train, krr_type, [krr_params[i]], save_dir,
                                                     model_type=('Ridge', KernelRidge()), scaler=False)
        else:
            krrs[i] = make_pipeline(StandardScaler(), KernelRidge(kernel=krr_type))
            krrs[i].fit(X_train, labels_train)
            print(f'Fitting done.')
        end = time.time()
        print(f"time to fit {krr_type} ridge : {str(end - start)}")

    # GRAPH KERNELS FIT
    if isinstance(train_dataset, GraphDataset):
        def compute_kernel(name, dataset, edge_to_node, normalize, wl_height=5):
            if name == 'wl':
                K = compute_wl_kernel(dataset, wl_height=wl_height, edge_to_node=edge_to_node,
                                      normalize=normalize)
            elif name == 'sp':
                K = compute_sp_kernel(dataset, normalize=normalize, edge_to_node=edge_to_node)
            elif name == 'mslap':
                K = compute_mslap_kernel(dataset, normalize=normalize, edge_to_node=edge_to_node)
            elif name == 'hadcode':
                K = compute_hadcode_kernel(dataset, normalize=normalize, edge_to_node=edge_to_node)
            else:
                assert False, f'unknown graph kernel: {graph_kernel}'
            return K

        wl_height = 5
        edge_to_node = True
        normalize = False
        # graph_kernel_names = ['wl', 'sp', 'hadcode']
        # graph_kernel_names = ['wl', 'sp']
        graph_kernel_names = []
        graph_kernels = []
        graph_krr_params = []
        for graph_kernel in graph_kernel_names:
            K = compute_kernel(graph_kernel, train_dataset, edge_to_node=edge_to_node, normalize=normalize,
                               wl_height=wl_height)
            graph_kernels.append(('precomputed', K, graph_kernel))
            graph_krr_params.append({'Ridge__kernel': ['precomputed'], 'Ridge__alpha': np.logspace(-5, 2, 11)})

        graph_krrs = [None] * len(graph_kernels)
        for i, (krr_type, K, name) in enumerate(graph_kernels):
            start = time.time()
            if fithyperparam:
                graph_krrs[i] = learn_or_load_modelhyperparams(K, labels_train, name, [graph_krr_params[i]],
                                                               save_dir,
                                                               model_type=('Ridge', KernelRidge()), scaler=False)
            else:
                graph_krrs[i] = make_pipeline(StandardScaler(), KernelRidge(kernel=name))
                graph_krrs[i].fit(K, labels_train)
                print(f'Fitting done.')
            end = time.time()
            print(f"time to fit {krr_type} ridge : {str(end - start)}")

    if isinstance(train_dataset, GraphDataset):
        n_permutation = 10
        our_r2_scores = []
        our_mse_scores = []
        our_mae_scores = []
        r2_scores_train = []
        mse_scores_train = []
        mae_scores_train = []
        ridge_r2_scores = []
        ridge_mse_scores = []
        ridge_mae_scores = []
        krr_scores = []
        for krr in krrs:
            krr_scores.append([])
        krr_graph_scores = []
        for graph_krr in graph_krrs:
            krr_graph_scores.append([])

        for n_perm in range(n_permutation):
            val_dataset.permute_graphs_in_dataset()
            # OUR APPROACH EVALUATION
            if model is not None:
                val_loader = val_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

                start = time.time()
                val_inZ = []
                elabels = []
                with torch.no_grad():
                    for j, data in enumerate(val_loader):
                        inp, labels = data
                        inp = val_dataset.format_data(inp, device)
                        labels = labels.to(device)
                        log_p, distloss, logdet, out = model(inp, labels)
                        val_inZ.append(out.detach().cpu().numpy())
                        elabels.append(labels.detach().cpu().numpy())
                val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(val_dataset), -1)
                elabels = np.concatenate(elabels, axis=0)
                end = time.time()
                print(f"time to get Z_val from X_val : {str(end - start)}")

                start = time.time()
                pred = zlinridge.predict(val_inZ)
                zridge_r2_score = zlinridge.score(val_inZ, elabels)
                zridge_mae_score = np.abs(pred - elabels).mean()
                zridge_mse_score = np.power(pred - elabels, 2).mean()
                # print(f'Our approach R2: {zridge_r2_score}, MSE: {zridge_mse_score}, MAE: {zridge_mae_score}')
                our_r2_scores.append(zridge_r2_score)
                our_mse_scores.append(zridge_mse_score)
                our_mae_scores.append(zridge_mae_score)
                end = time.time()
                print(f"time to predict with zlinridge : {str(end - start)}")

                # See on train
                start = time.time()
                pred = zlinridge.predict(Z)
                t_zridge_r2_score = zlinridge.score(Z, tlabels)
                t_zridge_mae_score = np.abs(pred - tlabels).mean()
                t_zridge_mse_score = np.power(pred - tlabels, 2).mean()
                # print(
                #     f'(On train) Our approach R2: {t_zridge_r2_score}, MSE: {t_zridge_mse_score}, MAE: {t_zridge_mae_score}')
                r2_scores_train.append(t_zridge_r2_score)
                mse_scores_train.append(t_zridge_mse_score)
                mae_scores_train.append(t_zridge_mae_score)
                end = time.time()
                print(f"time to predict with zlinridge (on train) : {str(end - start)}")

            # KERNELS EVALUATION
            X_val = val_dataset.get_flattened_X()
            # X_val = val_dataset.X.reshape(val_dataset.X.shape[0], -1)
            labels_val = val_dataset.true_labels

            start = time.time()
            ridge_r2_score = linridge.score(X_val, labels_val)
            ridge_mae_score = np.abs(linridge.predict(X_val) - labels_val).mean()
            ridge_mse_score = np.power(linridge.predict(X_val) - labels_val, 2).mean()
            ridge_r2_scores.append(ridge_r2_score)
            ridge_mae_scores.append(ridge_mae_score)
            ridge_mse_scores.append(ridge_mse_score)
            end = time.time()
            print(f"time to predict with xlinridge : {str(end - start)}")

            start = time.time()
            for i, krr in enumerate(krrs):
                krr_r2_score = krr.score(X_val, labels_val)
                krr_mae_score = np.abs(krr.predict(X_val) - labels_val).mean()
                krr_mse_score = np.power(krr.predict(X_val) - labels_val, 2).mean()
                # krr_score = np.abs(krr.predict(X_val) - labels_val).mean()
                krr_scores[i].append((krr_r2_score, krr_mae_score, krr_mse_score))
            end = time.time()
            print(f"time to predict with {len(krrs)} kernelridge : {str(end - start)}")

            start = time.time()
            # GRAPH KERNELS EVALUATION
            for i, graph_krr in enumerate(graph_krrs):
                K_val = compute_kernel(graph_kernels[i][2], (val_dataset, train_dataset), edge_to_node=edge_to_node,
                                       normalize=normalize, wl_height=wl_height)
                # K_val = compute_wl_kernel((val_dataset, train_dataset), wl_height=wl_height, edge_to_node=edge_to_node,
                #                           normalize=normalize)
                graph_krr_r2_score = graph_krr.score(K_val, labels_val)
                graph_krr_mae_score = np.abs(graph_krr.predict(K_val) - labels_val).mean()
                graph_krr_mse_score = np.power(graph_krr.predict(K_val) - labels_val, 2).mean()

                # print(f'GraphKernelRidge ({graph_kernels[i][2]}) R2: {graph_krr_r2_score}, '
                #       f'MSE: {graph_krr_mse_score}, MAE: {graph_krr_mae_score}')
                krr_graph_scores[i].append((graph_krr_r2_score, graph_krr_mae_score, graph_krr_mse_score))
            end = time.time()
            print(f"time to predict with {len(graph_krrs)} graphkernelridge : {str(end - start)}")

        # PRINT RESULTS
        lines = []
        print('Predictions scores :')
        r2_mean_score = np.mean(ridge_r2_scores)
        mse_mean_score = np.mean(ridge_mse_scores)
        mae_mean_score = np.mean(ridge_mae_scores)
        r2_std_score = np.std(ridge_r2_scores)
        mse_std_score = np.std(ridge_mse_scores)
        mae_std_score = np.std(ridge_mae_scores)
        score_str = f'Ridge R2: {ridge_r2_scores}, MSE: {ridge_mse_scores}, MAE: {ridge_mae_scores} \n' \
                    f'Mean Scores: R2: {r2_mean_score}, MSE: {mse_mean_score}, MAE: {mae_mean_score} \n' \
                    f'Std Scores: R2: {r2_std_score}, MSE: {mse_std_score}, MAE: {mae_std_score}'
        print(score_str)
        lines += [score_str, '\n']

        for j, krr_type in enumerate(krr_types):
            r2_scores = []
            mse_scores = []
            mae_scores = []
            for krr_r2_score, krr_mae_score, krr_mse_score in krr_scores[j]:
                r2_scores.append(krr_r2_score)
                mse_scores.append(krr_mse_score)
                mae_scores.append(krr_mae_score)
            r2_mean_score = np.mean(r2_scores)
            mse_mean_score = np.mean(mse_scores)
            mae_mean_score = np.mean(mae_scores)
            r2_std_score = np.std(r2_scores)
            mse_std_score = np.std(mse_scores)
            mae_std_score = np.std(mae_scores)
            score_str = f'KernelRidge ({krr_type}) R2: {r2_scores}, MSE: {mse_scores}, MAE: {mae_scores} \n' \
                        f'Mean Scores: R2: {r2_mean_score}, MSE: {mse_mean_score}, MAE: {mae_mean_score} \n' \
                        f'Std Scores: R2: {r2_std_score}, MSE: {mse_std_score}, MAE: {mae_std_score}'
            print(score_str)
            lines += [score_str, '\n']

        for j, graph_krr in enumerate(graph_krrs):
            r2_scores = []
            mse_scores = []
            mae_scores = []
            for graph_krr_r2_score, graph_krr_mae_score, graph_krr_mse_score in krr_graph_scores[j]:
                r2_scores.append(graph_krr_r2_score)
                mse_scores.append(graph_krr_mse_score)
                mae_scores.append(graph_krr_mae_score)
            r2_mean_score = np.mean(r2_scores)
            mse_mean_score = np.mean(mse_scores)
            mae_mean_score = np.mean(mae_scores)
            r2_std_score = np.std(r2_scores)
            mse_std_score = np.std(mse_scores)
            mae_std_score = np.std(mae_scores)
            score_str = f'GraphKernelRidge ({graph_kernels[j][2]}) R2: {r2_scores}, MSE: {mse_scores}, MAE: {mae_scores} \n' \
                        f'Mean Scores: R2: {r2_mean_score}, MSE: {mse_mean_score}, MAE: {mae_mean_score} \n' \
                        f'Std Scores: R2: {r2_std_score}, MSE: {mse_std_score}, MAE: {mae_std_score}'
            print(score_str)
            lines += [score_str, '\n']

        r2_mean_score = np.mean(our_r2_scores)
        mse_mean_score = np.mean(our_mse_scores)
        mae_mean_score = np.mean(our_mae_scores)
        r2_std_score = np.std(our_r2_scores)
        mse_std_score = np.std(our_mse_scores)
        mae_std_score = np.std(our_mae_scores)
        score_str = f'Our approach R2: {our_r2_scores}, MSE: {our_mse_scores}, MAE: {our_mae_scores} \n' \
                    f'Mean Scores: R2: {r2_mean_score}, MSE: {mse_mean_score}, MAE: {mae_mean_score} \n' \
                    f'Std Scores: R2: {r2_std_score}, MSE: {mse_std_score}, MAE: {mae_std_score}'
        print(score_str)
        lines += [score_str, '\n']
        r2_mean_score = np.mean(r2_scores_train)
        mse_mean_score = np.mean(mse_scores_train)
        mae_mean_score = np.mean(mae_scores_train)
        r2_std_score = np.std(r2_scores_train)
        mse_std_score = np.std(mse_scores_train)
        mae_std_score = np.std(mae_scores_train)
        score_str = f'(On train) Our approach R2: {r2_scores_train}, MSE: {mse_scores_train}, MAE: {mae_scores_train} \n' \
                    f'Mean Scores: R2: {r2_mean_score}, MSE: {mse_mean_score}, MAE: {mae_mean_score} \n' \
                    f'Std Scores: R2: {r2_std_score}, MSE: {mse_std_score}, MAE: {mae_std_score}'
        print(score_str)
        lines += [score_str, '\n']

    else:
        # OUR APPROACH EVALUATION
        if model is not None:
            val_loader = val_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

            start = time.time()
            val_inZ = []
            elabels = []
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    inp, labels = data
                    inp = val_dataset.format_data(inp, device)
                    labels = labels.to(device)
                    log_p, distloss, logdet, out = model(inp, labels)
                    val_inZ.append(out.detach().cpu().numpy())
                    elabels.append(labels.detach().cpu().numpy())
            val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(val_dataset), -1)
            elabels = np.concatenate(elabels, axis=0)
            end = time.time()
            print(f"time to get Z_val from X_val : {str(end - start)}")

            pred = zlinridge.predict(val_inZ)
            zridge_r2_score = zlinridge.score(val_inZ, elabels)
            zridge_mae_score = np.abs(pred - elabels).mean()
            zridge_mse_score = np.power(pred - elabels, 2).mean()
            q2_ext = 1 - (np.sum(np.power(elabels - pred, 2)) / elabels.shape[0]) / (
                    np.sum(np.power(tlabels - np.mean(tlabels), 2)) / tlabels.shape[0])
            print(
                f'Our approach R2: {zridge_r2_score}, MSE: {zridge_mse_score}, MAE: {zridge_mae_score}, q2_ext: {q2_ext}')

            # See on train
            pred = zlinridge.predict(Z)
            t_zridge_r2_score = zlinridge.score(Z, tlabels)
            t_zridge_mae_score = np.abs(pred - tlabels).mean()
            t_zridge_mse_score = np.power(pred - tlabels, 2).mean()
            print(
                f'(On train) Our approach R2: {t_zridge_r2_score}, MSE: {t_zridge_mse_score}, MAE: {t_zridge_mae_score}')

            # TEST project between the means
            means = model.means.detach().cpu().numpy()
            proj, dot_val = project_between(val_inZ, means[0], means[1])
            # pred = ((proj - means[1]) / (means[0] - means[1])) * (
            #         model.label_max - model.label_min) + model.label_min
            # pred = pred.mean(axis=1)
            pred = dot_val.squeeze() * (model.label_max - model.label_min) + model.label_min
            projection_mse_score = np.power((pred - elabels), 2).mean()
            projection_mae_score = np.abs((pred - elabels)).mean()
            print(f'Our approach (projection) MSE: {projection_mse_score}, MAE: {projection_mae_score}')

        # KERNELS EVALUATION
        X_val = val_dataset.get_flattened_X()
        # X_val = val_dataset.X.reshape(val_dataset.X.shape[0], -1)
        labels_val = val_dataset.true_labels

        krr_scores = []
        for krr in krrs:
            krr_scores.append([])

        ridge_r2_score = linridge.score(X_val, labels_val)
        ridge_mae_score = np.abs(linridge.predict(X_val) - labels_val).mean()
        ridge_mse_score = np.power(linridge.predict(X_val) - labels_val, 2).mean()

        for i, krr in enumerate(krrs):
            krr_r2_score = krr.score(X_val, labels_val)
            krr_mae_score = np.abs(krr.predict(X_val) - labels_val).mean()
            krr_mse_score = np.power(krr.predict(X_val) - labels_val, 2).mean()
            # krr_score = np.abs(krr.predict(X_val) - labels_val).mean()
            krr_scores[i].append((krr_r2_score, krr_mae_score, krr_mse_score))

        lines = []
        print('Predictions scores :')
        score_str = f'Ridge R2: {ridge_r2_score}, MSE: {ridge_mse_score}, MAE: {ridge_mae_score}'
        print(score_str)
        lines += [score_str, '\n']
        for j, krr_type in enumerate(krr_types):
            krr_r2_score, krr_mae_score, krr_mse_score = krr_scores[j][0]
            score_str = f'KernelRidge ({krr_type}) R2: {krr_r2_score}, MSE: {krr_mse_score}, MAE: {krr_mae_score}'
            print(score_str)
            lines += [score_str, '\n']

        score_str = f'Our approach (projection) MSE: {projection_mse_score}, MAE: {projection_mae_score}'
        print(score_str)
        lines += [score_str, '\n']
        score_str = f'Our approach R2: {zridge_r2_score}, MSE: {zridge_mse_score}, MAE: {zridge_mae_score}, q2_ext: {q2_ext}'
        print(score_str)
        lines += [score_str, '\n']
        score_str = f'(On train) Our approach R2: {t_zridge_r2_score}, MSE: {t_zridge_mse_score}, MAE: {t_zridge_mae_score}'
        print(score_str)
        lines += [score_str, '\n']

    with open(f"{save_dir}/eval_res.txt", 'w') as f:
        f.writelines(lines)

    return zlinridge


def create_figures_XZ(model, train_dataset, save_path, device, std_noise=0.1, only_Z=False, batch_size=20):
    size_pt_fig = 5

    loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    X = []
    Z = []
    tlabels = []
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(loader):
            inp, labels = data
            inp = train_dataset.format_data(inp, device)
            if isinstance(inp, list):
                for i in range(len(inp)):
                    inp[i] = (inp[i] + torch.from_numpy(np.random.randn(*inp[i].shape)).float().to(device) * std_noise)
            else:
                inp = inp + torch.from_numpy(np.random.randn(*inp.shape)).float().to(device) * std_noise
            labels = labels.to(device)
            # inp = (data[0] + np.random.randn(*data[0].shape) * std_noise).float().to(device)
            # labels = data[1].to(device)
            log_p, distloss, logdet, out = model(inp, labels)
            if not only_Z:
                X.append(inp.detach().cpu().numpy())
            Z.append(out.detach().cpu().numpy())
            tlabels.append(labels.detach().cpu().numpy())
    tlabels = np.concatenate(tlabels, axis=0)
    if not only_Z:
        X = np.concatenate(X, axis=0).reshape(len(train_dataset), -1)
        save_fig(X, tlabels, size=size_pt_fig, save_path=f'{save_path}/X_space')
    else:
        X = None

    Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)
    save_fig(Z, tlabels, size=size_pt_fig, save_path=f'{save_path}/Z_space')

    # PCA
    pca = PCA(n_components=2)
    pca.fit(Z)
    pca_Z = pca.transform(Z)
    save_fig(pca_Z, tlabels, size=size_pt_fig, save_path=f'{save_path}/PCA_Z_space')
    return X, Z


def evaluate_preimage(model, val_dataset, device, save_dir, print_as_mol=True, print_as_graph=True,
                      eval_type='regression', batch_size=20, means=None):
    assert eval_type in ['regression', 'classification'], 'unknown pre-image generation evaluation type'

    if means is None:
        model_means = model.means.detach().cpu().numpy()
    else:
        model_means = means

    if eval_type == 'regression':
        y_min = model.label_min
        y_max = model.label_max
        samples = []
        # true_X = val_dataset.X
        true_X = val_dataset.get_flattened_X(with_added_features=True)
        for i, y in enumerate(val_dataset.true_labels):
            # mean, cov = model.get_regression_gaussian_sampling_parameters(y)
            alpha_y = (y - y_min) / (y_max - y_min)
            mean = alpha_y * model_means[0] + (1 - alpha_y) * model_means[1]
            samples.append(np.expand_dims(mean, axis=0))
    else:
        samples = []
        # true_X = val_dataset.X
        true_X = val_dataset.get_flattened_X(with_added_features=True)
        for i, y in enumerate(val_dataset.true_labels):
            mean = model_means[y]
            samples.append(np.expand_dims(mean, axis=0))

    samples = np.concatenate(samples, axis=0)

    z_shape = model.calc_last_z_shape(val_dataset.in_size)
    nb_batch = math.ceil(samples.shape[0] / batch_size)
    all_res = []
    samples = torch.from_numpy(samples)
    with torch.no_grad():
        for j in range(nb_batch):
            size = samples[j * batch_size:(j + 1) * batch_size].shape[0]
            input = samples[j * batch_size:(j + 1) * batch_size].reshape(size, *z_shape).float().to(device)

            res = model.reverse(input)
            all_res.append(res.detach().cpu().numpy())
    all_res = np.concatenate(all_res, axis=0)

    distance = np.mean(np.power(all_res - true_X, 2))
    print(f'Pre-image distance :{distance}')

    # For graphs
    if isinstance(val_dataset, GraphDataset):
        # x_shape = val_dataset.X[0][0].shape
        # adj_shape = val_dataset.X[0][1].shape
        x_shape, adj_shape = val_dataset.get_input_shapes()
        x_sh = x_shape[0]
        for v in x_shape[1:]:
            x_sh *= v
        x = all_res[:, :x_sh].reshape(all_res.shape[0], *x_shape)
        adj = all_res[:, x_sh:].reshape(all_res.shape[0], *adj_shape)

        # if feature has been added
        if val_dataset.add_feature is not None and val_dataset.add_feature > 0:
            af = val_dataset.add_feature
            x = x[:, :, :-af]

        # if mols
        if print_as_mol and val_dataset.atomic_num_list is not None:
            from utils.graphs.mol_utils import check_validity, save_mol_png
            atomic_num_list = val_dataset.atomic_num_list
            valid_mols = check_validity(adj, x, atomic_num_list)['valid_mols']
            mol_dir = os.path.join(save_dir, 'generated_means')
            os.makedirs(mol_dir, exist_ok=True)
            for ind, mol in enumerate(valid_mols):
                save_mol_png(mol, os.path.join(mol_dir, '{}.png'.format(ind)))

        # if graphs
        if print_as_graph:
            # define the colormap
            cmap = plt.cm.jet
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            n_type = x.shape[-1] - 1
            node_colors = [cmaplist[i * math.floor(len(cmaplist) / n_type)] for i in range(0, n_type)]

            graphs = val_dataset.get_full_graphs(data=list(zip(x, adj)),
                                                 attributed_node=val_dataset.is_attributed_node_dataset())
            if val_dataset.label_map is None:
                inv_map = {i + 1: str(i) for i in range(x.shape[-1])}
            else:
                inv_map = {v: k for k, v in val_dataset.label_map.items()}
            for i, graph in enumerate(graphs):
                if graph is None:
                    continue
                path = f'{save_dir}/generated_means_graphs/'
                os.makedirs(path, exist_ok=True)
                title = '\^y:' + str(round(val_dataset.true_labels[i], 2))
                if val_dataset.is_attributed_node_dataset():
                    save_nx_graph_attr(graph, save_path=f'{path}/{str(i).zfill(4)}', title=title)
                else:
                    save_nx_graph(graph, inv_map, save_path=f'{path}/{str(i).zfill(4)}', title=title,
                                  n_atom_type=x.shape[-1] - 1,
                                  colors=node_colors)


def evaluate_preimage2(model, val_dataset, device, save_dir, n_y=50, n_samples_by_y=20, print_as_mol=True,
                       print_as_graph=True, eval_type='regression', batch_size=20, predmodel=None, means=None,
                       debug=False):
    assert eval_type in ['regression', 'classification'], 'unknown pre-image generation evaluation type'

    if means is None:
        model_means = model.means.detach().cpu().numpy()
    else:
        model_means = means

    # covariance_matrix = construct_covariance(model.eigvecs.cpu()[0].squeeze(), model.eigvals.cpu()[0].squeeze())
    # covariance_matrices = model.covariance_matrices
    covariance_matrices = [eigval * np.identity(eigval.shape[0]) for _, eigval, _ in model.gp]
    # covariance_matrices = [0.1 * np.identity(eigval.shape[0]) for _, eigval, _ in model.gp]
    # TEST smaller variance
    # for cov in covariance_matrices:
    #     cov[np.where(cov > 1)] = 1
    samples = []
    datasets_close_samples = []
    datasets_close_y = []
    import scipy
    n_closest = 10

    if eval_type == 'regression':
        y_min = model.label_min
        y_max = model.label_max
        ys = np.linspace(y_min, y_max, n_y)
    else:
        ys = np.unique(val_dataset.true_labels)
    sampled_ys = np.concatenate([[y] * n_samples_by_y for y in ys])

    close_samples_dist = scipy.spatial.distance.cdist(np.expand_dims(ys, axis=1),
                                                      np.expand_dims(val_dataset.true_labels, axis=1))
    close_samples_idx = np.argsort(close_samples_dist, axis=1)

    pred_ys = []
    if eval_type == 'regression':
        for i, y in enumerate(ys):
            # mean, cov = model.get_regression_gaussian_sampling_parameters(y)
            alpha_y = (y - y_min) / (y_max - y_min)
            mean = alpha_y * model_means[0] + (1 - alpha_y) * model_means[1]
            covariance_matrix = covariance_matrices[0]
            z = np.random.multivariate_normal(mean, covariance_matrix, n_samples_by_y)
            # samples.append(np.expand_dims(mean, axis=0))
            samples.append(z)

            if predmodel is not None:
                pred_ys.append(predmodel.predict(z))

            # Get close samples in dataset w.r.t the y property
            datasets_close_samples.append(np.array(val_dataset.X)[close_samples_idx[i, :n_closest]])
            datasets_close_y.append(val_dataset.true_labels[close_samples_idx[i, :n_closest]])
    else:
        for i, y in enumerate(ys):
            mean = model_means[y]
            z = np.random.multivariate_normal(mean, covariance_matrices[y], n_samples_by_y)
            samples.append(z)

            if predmodel is not None:
                pred_ys.append(predmodel.predict(z))

            # Get close samples in dataset w.r.t the y property
            datasets_close_samples.append(np.array(val_dataset.X)[close_samples_idx[i, :n_closest]])
            datasets_close_y.append(val_dataset.true_labels[close_samples_idx[i, :n_closest]])
    if predmodel is not None:
        pred_ys = np.concatenate(pred_ys, axis=0)
    samples = np.concatenate(samples, axis=0)
    datasets_close_samples = np.concatenate([np.expand_dims(samples, 0) for samples in datasets_close_samples],
                                            axis=0).transpose(0, 2, 1)

    z_shape = model.calc_last_z_shape(val_dataset.in_size)
    nb_batch = math.ceil(samples.shape[0] / batch_size)
    all_res = []
    samples = torch.from_numpy(samples)
    with torch.no_grad():
        for j in range(nb_batch):
            size = samples[j * batch_size:(j + 1) * batch_size].shape[0]
            input = samples[j * batch_size:(j + 1) * batch_size].reshape(size, *z_shape).float().to(device)

            if debug:
                create_folder(f"{save_dir}/flows_gen/{j}")
                res = model.reverse_debug(input, val_dataset, save_dir=f"{save_dir}/flows_gen/{j}")
            else:
                res = model.reverse(input)
            all_res.append(res.detach().cpu().numpy())
    all_res = np.concatenate(all_res, axis=0)

    if val_dataset.get_n_dim() == 2:
        save_fig(all_res, ys, size=5, save_path=f'{save_dir}/pre_images_inX', eps=True)
        save_fig(samples.detach().cpu().numpy(), ys, size=5, save_path=f'{save_dir}/pre_images_inZ', eps=True)

    # For graphs
    if isinstance(val_dataset, GraphDataset):
        # x_shape = val_dataset.X[0][0].shape
        # adj_shape = val_dataset.X[0][1].shape
        x_shape, adj_shape = val_dataset.get_input_shapes()
        x_sh = x_shape[0]
        for v in x_shape[1:]:
            x_sh *= v
        x = all_res[:, :x_sh].reshape(all_res.shape[0], *x_shape)
        adj = all_res[:, x_sh:].reshape(all_res.shape[0], *adj_shape)

        # if feature has been added
        if val_dataset.add_feature is not None and val_dataset.add_feature > 0:
            af = val_dataset.add_feature
            x = x[:, :, :-af]

        # if mols
        if print_as_mol and val_dataset.atomic_num_list is not None:
            from utils.graphs.mol_utils import check_validity, save_mol_png
            atomic_num_list = val_dataset.atomic_num_list
            results_g = check_validity(adj, x, atomic_num_list, with_idx=True)
            valid_mols = results_g['valid_mols']
            valid_smiles = results_g['valid_smiles']
            idxs_valid = results_g['idxs']
            mol_dir = os.path.join(save_dir, 'generated_samples')
            os.makedirs(mol_dir, exist_ok=True)

            psize = (200, 200)
            from rdkit import Chem, DataStructs
            from rdkit.Chem import Draw, AllChem
            from ordered_set import OrderedSet
            for ind, mol in enumerate(valid_mols):
                # save_mol_png(mol, os.path.join(mol_dir, '{}.png'.format(ind)))

                # show with closest dataset samples
                mol_idx = idxs_valid[ind]
                y_idx = math.floor(mol_idx / n_samples_by_y)
                close_samples_x = datasets_close_samples[y_idx][0]
                close_samples_adj = datasets_close_samples[y_idx][1]
                close_y = datasets_close_y[y_idx]
                c_x = np.concatenate([np.expand_dims(v, axis=0) for v in close_samples_x], axis=0)
                c_adj = np.concatenate([np.expand_dims(v, axis=0) for v in close_samples_adj], axis=0)
                # results = check_validity(c_adj, c_x, atomic_num_list, return_unique=False, debug=False,
                #                          custom_bond_assignement=custom_bond_assignement,
                #                          virtual_bond_idx=virtual_bond_idx)
                results = check_validity(c_adj, c_x, atomic_num_list, return_unique=False, debug=False)
                close_mols = results['valid_mols']
                close_smiles = results['valid_smiles']
                with_seeds = [mol] + close_mols
                all_ys = [ys[y_idx]] + list(close_y)
                legends_with_seed = [valid_smiles[ind]] + close_smiles
                legends_with_seed[0] = legends_with_seed[0] + ', \^y:' + str(round(all_ys[0], 2))
                if predmodel is not None:
                    pred_y = pred_ys[mol_idx]
                    legends_with_seed[0] = legends_with_seed[0] + ', pred_y:' + str(round(pred_y, 2))
                legends_with_seed[1:] = [smile + ', y:' + str(round(y, 2)) for y, smile in
                                         zip(all_ys[1:], legends_with_seed[1:])]
                img = Draw.MolsToGridImage(with_seeds, legends=legends_with_seed,
                                           molsPerRow=int((n_closest + 1) / 3),
                                           subImgSize=psize)

                img.save(f'{save_dir}/generated_samples/{ind}_close_grid.png')

            # save image of generated mols
            legends_with_seed = valid_smiles
            ys_idx = np.floor(np.array(idxs_valid) / n_samples_by_y).astype(int)
            # legends_with_seed = [legend + ', y:' + str(round(ys[ys_idx[i]], 2)) for i, legend in enumerate(legends_with_seed)]
            legends_with_seed = ['y=' + str(round(ys[ys_idx[i]], 2)) for i, legend in enumerate(legends_with_seed)]
            if predmodel is not None:
                legends_with_seed = [legend + ', f(G)=' + str(round(pred_ys[i], 2)) for i, legend in
                                     enumerate(legends_with_seed)]
            img = Draw.MolsToGridImage(valid_mols, legends=legends_with_seed,
                                       # molsPerRow=int(2* math.sqrt(len(valid_mols))),
                                       molsPerRow=6,
                                       subImgSize=psize, useSVG=True)
            with open(f'{save_dir}/generated_samples/all_generated_grid.svg', 'w') as f:
                f.write(img)
            # img.save(f'{save_dir}/generated_samples/all_generated_grid.png')

            print('mean_error between y sampled and pred y:', np.mean(sampled_ys - pred_ys))

        # if graphs
        if print_as_graph:

            # define the colormap
            cmap = plt.cm.jet
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            n_type = x.shape[-1] - 1
            node_colors = [cmaplist[i * math.floor(len(cmaplist) / n_type)] for i in range(0, n_type)]

            close_xadj = [(np.concatenate([np.expand_dims(v, axis=0) for v in samples[0]], axis=0),
                           np.concatenate([np.expand_dims(v, axis=0) for v in samples[1]], axis=0)) for samples in
                          list(datasets_close_samples)]
            close_graphs = []
            for close_xadj_set in close_xadj:
                close_graphs.append(val_dataset.get_full_graphs(data=list(zip(*close_xadj_set)),
                                                                attributed_node=val_dataset.is_attributed_node_dataset()))

            graphs = val_dataset.get_full_graphs(data=list(zip(x, adj)),
                                                 attributed_node=val_dataset.is_attributed_node_dataset())
            if val_dataset.label_map is None:
                inv_map = {i + 1: str(i) for i in range(x.shape[-1])}
            else:
                inv_map = {v: k for k, v in val_dataset.label_map.items()}
            for i, graph in enumerate(graphs):
                if graph is None:
                    continue
                graph_idx = i
                y_idx = math.floor(graph_idx / n_samples_by_y)
                close_y = datasets_close_y[y_idx]
                all_ys = [ys[y_idx]] + list(close_y)

                path = f'{save_dir}/generated_samples_graphs/{str(i).zfill(4)}'
                os.makedirs(path, exist_ok=True)
                title = '\^y:' + str(round(all_ys[0], 2))
                if predmodel is not None:
                    pred_y = pred_ys[graph_idx]
                    title = title + ', pred_y:' + str(round(pred_y, 2))
                if val_dataset.is_attributed_node_dataset():
                    save_nx_graph_attr(graph, save_path=f'{path}/gen_{str(i).zfill(4)}', title=title)
                else:
                    save_nx_graph(graph, inv_map, save_path=f'{path}/gen_{str(i).zfill(4)}', title=title,
                                  n_atom_type=x.shape[-1] - 1,
                                  colors=node_colors)

                # closest graphs
                close_graphs_i = close_graphs[y_idx]
                for j, close_graph in enumerate(close_graphs_i):
                    title = 'y:' + str(round(all_ys[j], 2))
                    if val_dataset.is_attributed_node_dataset():
                        save_nx_graph_attr(close_graph, save_path=f'{path}/{str(j).zfill(4)}_{str(i).zfill(4)}',
                                           title=title)
                    else:
                        save_nx_graph(close_graph, inv_map, save_path=f'{path}/{str(j).zfill(4)}_{str(i).zfill(4)}',
                                      title=title,
                                      n_atom_type=x.shape[-1] - 1, colors=node_colors)
                # create grid
                nrow = math.ceil(math.sqrt(n_closest))
                images = organise_data(path, nrow=nrow)
                res_name = f'{str(i).zfill(4)}_close_grid'
                format(images, nrow=nrow, save_path=f'{save_dir}/generated_samples_graphs', res_name=res_name)


def evaluate_image_interpolations(model, val_dataset, device, save_dir, n_sample=20, n_interpolation=20, label=None,
                                  debug=False):
    def get_PIL_image_RGB(dataset, idx):
        im = dataset.X[idx].transpose(1, 2, 0)
        im = Image.fromarray(im.squeeze(), mode='RGB')
        return im

    def get_PIL_image_L(dataset, idx):
        im = dataset.X[idx]
        im = Image.fromarray(im.squeeze(), mode='L')
        return im

    def get_PIL_image_1b(dataset, idx):
        im = dataset.X[idx]
        im = Image.fromarray(im.squeeze())
        return im

    get_PIL_image = get_PIL_image_L if val_dataset.n_channel == 1 else get_PIL_image_RGB
    get_PIL_image = get_PIL_image_1b if val_dataset.dataset_name == 'olivetti_faces' else get_PIL_image

    def get_data(dataset, idx):
        target = int(dataset.true_labels[idx])

        img = get_PIL_image(dataset, idx)
        if dataset.transform is not None:
            img = dataset.transform(img)

        return img, target

    if label is not None:
        dset_labels = np.array([label])
    else:
        dset_labels = np.unique(val_dataset.true_labels)
    done = []
    all_res = []
    with torch.no_grad():
        for j in range(n_sample):
            idxs = np.where(val_dataset.true_labels == dset_labels[j % dset_labels.shape[0]])[0]
            i_pt0 = random.choice(idxs)
            while i_pt0 in done:
                i_pt0 = random.choice(idxs)
            done.append(i_pt0)
            # i_pt0 = np.random.randint(0, len(val_dataset))
            i_pt1 = i_pt0
            while val_dataset.true_labels[i_pt1] != val_dataset.true_labels[i_pt0] or i_pt0 == i_pt1 or i_pt1 in done:
                i_pt1 = np.random.randint(0, len(val_dataset))
            pt0, y0 = get_data(val_dataset, i_pt0)
            pt1, y1 = get_data(val_dataset, i_pt1)
            inp = torch.from_numpy(np.concatenate([np.expand_dims(pt0, axis=0), np.expand_dims(pt1, axis=0)]))
            inp = val_dataset.format_data(inp, device)
            labels = torch.from_numpy(
                np.concatenate([np.expand_dims(y0, axis=0), np.expand_dims(y1, axis=0)], axis=0)).to(
                device)

            log_p, distloss, logdet, out = model(inp, labels)

            d = out[1] - out[0]
            z_list = [(out[0] + i * 1.0 / (n_interpolation + 1) * d).unsqueeze(0) for i in range(n_interpolation + 2)]

            z_array = torch.cat(z_list, dim=0)

            if debug:
                create_folder(f'{save_dir}/test_generation/test_flows_gen/{j}')
                res = model.reverse_debug(z_array, val_dataset,
                                          save_dir=f"{save_dir}/test_generation/test_flows_gen/{j}")
            else:
                res = model.reverse(z_array)
            all_res.append(res.detach().cpu().numpy())

    # to_remove = [0, 2, 6, 11, 12, 13, 14]
    # for i in range(len(to_remove)):
    #     del all_res[to_remove[-1 - i]]

    all_res = np.concatenate(all_res, axis=0)

    if isinstance(val_dataset, ImDataset):
        images = val_dataset.rescale(all_res)
        utils.save_image(
            torch.from_numpy(images),
            f"{save_dir}/test_generation/interpolations.png",
            normalize=True,
            nrow=n_interpolation + 2,
            range=(0, 255),
        )


def evaluate_graph_interpolations(model, val_dataset, device, save_dir, n_sample=20, n_interpolation=20, Z=None,
                                  print_as_mol=True, print_as_graph=True, eval_type='regression', label=None):
    # si Z est donné, PCA sur 2 dimensions
    if Z is not None:
        pca = PCA(n_components=2)
        pca.fit(Z)
        pca_Z = pca.transform(Z)

        create_folder(f'{save_dir}/generated_interp')

    def get_data(dataset, idx):
        sample = dataset.X[idx]
        y = dataset.true_labels[idx]
        if dataset.transform:
            sample = dataset.transform(*sample)
        return sample, y

    if eval_type == 'regression':
        y_min = model.label_min
        y_max = model.label_max
        y_margin = (y_max - y_min) / 3
    else:
        done = []
        if label is not None:
            dset_labels = np.array([label])
        else:
            dset_labels = np.unique(val_dataset.true_labels)

    all_res = []
    with torch.no_grad():
        for j in range(n_sample):
            if eval_type == 'regression':
                i_pt0 = np.random.randint(0, len(val_dataset))
                i_pt1 = i_pt0
                while abs(val_dataset.true_labels[i_pt1] - val_dataset.true_labels[i_pt0]) < y_margin:
                    i_pt1 = np.random.randint(0, len(val_dataset))
            else:
                idxs = np.where(val_dataset.true_labels == dset_labels[j % dset_labels.shape[0]])[0]
                i_pt0 = random.choice(idxs)
                while i_pt0 in done:
                    i_pt0 = random.choice(idxs)
                done.append(i_pt0)
                i_pt1 = i_pt0
                while val_dataset.true_labels[i_pt1] != val_dataset.true_labels[
                    i_pt0] or i_pt0 == i_pt1 or i_pt1 in done:
                    i_pt1 = np.random.randint(0, len(val_dataset))

            pt0, y0 = get_data(val_dataset, i_pt0)
            pt1, y1 = get_data(val_dataset, i_pt1)
            inp = []
            for i in range(len(pt0)):
                inp.append(torch.from_numpy(
                    np.concatenate([np.expand_dims(pt0[i], axis=0), np.expand_dims(pt1[i], axis=0)], axis=0)))
            inp = val_dataset.format_data(inp, device)
            labels = torch.from_numpy(
                np.concatenate([np.expand_dims(y0, axis=0), np.expand_dims(y1, axis=0)], axis=0)).to(
                device)

            log_p, distloss, logdet, out = model(inp, labels)

            d = out[1] - out[0]
            z_list = [(out[0] + i * 1.0 / (n_interpolation + 1) * d).unsqueeze(0) for i in range(n_interpolation + 2)]

            z_array = torch.cat(z_list, dim=0)

            res = model.reverse(z_array)
            all_res.append(res.detach().cpu().numpy())

            if Z is not None:
                pca_interp = pca.transform(z_array.detach().cpu().numpy())
                data = np.concatenate((pca_Z, pca_interp), axis=0)
                lab = np.zeros(data.shape[0])
                lab[-(2 + n_interpolation):] = 1
                save_fig(data, lab, size=5, save_path=f'{save_dir}/generated_interp/{str(j).zfill(4)}_res_pca')

    all_res = np.concatenate(all_res, axis=0)

    # for graphs
    if isinstance(val_dataset, GraphDataset):
        # x_shape = val_dataset.X[0][0].shape
        # adj_shape = val_dataset.X[0][1].shape
        x_shape, adj_shape = val_dataset.get_input_shapes()

        if print_as_graph:
            # define the colormap
            cmap = plt.cm.jet
            # extract all colors from the .jet map
            cmaplist = [cmap(i) for i in range(cmap.N)]
            n_type = x_shape[-1] - 1
            node_colors = [cmaplist[i * math.floor(len(cmaplist) / n_type)] for i in range(0, n_type)]

        x_sh = x_shape[0]
        for v in x_shape[1:]:
            x_sh *= v
        x = all_res[:, :x_sh].reshape(all_res.shape[0], *x_shape)
        adj = all_res[:, x_sh:].reshape(all_res.shape[0], *adj_shape)
        atomic_num_list = val_dataset.atomic_num_list

        # if feature has been added
        if val_dataset.add_feature is not None and val_dataset.add_feature > 0:
            af = val_dataset.add_feature
            x = x[:, :, :-af]

        # Interps
        for n in range(int(all_res.shape[0] / (n_interpolation + 2))):
            xm = x[n * (n_interpolation + 2):(n + 1) * (n_interpolation + 2)]
            adjm = adj[n * (n_interpolation + 2):(n + 1) * (n_interpolation + 2)]

            # if mols
            if print_as_mol and atomic_num_list is not None:
                interpolation_mols = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
                                      for x_elem, adj_elem in zip(xm, adjm)]
                valid_mols = [mol for mol in interpolation_mols if mol is not None]
                valid_mols_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]

                valid_mols_smiles_unique = list(OrderedSet(valid_mols_smiles))
                valid_mols_unique = [Chem.MolFromSmiles(s) for s in valid_mols_smiles_unique]
                valid_mols_smiles_unique_label = []

                mol0 = valid_mols[0]
                mol1 = valid_mols[-1]
                smile0 = valid_mols_smiles[0]
                smile1 = valid_mols_smiles[-1]
                fp0 = AllChem.GetMorganFingerprint(mol0, 2)

                for s, m in zip(valid_mols_smiles_unique, valid_mols_unique):
                    fp = AllChem.GetMorganFingerprint(m, 2)
                    sim = DataStructs.TanimotoSimilarity(fp, fp0)
                    s = '{:.2f}\n'.format(sim) + s
                    if s == smile0:
                        s = '***[' + s + ']***'
                    valid_mols_smiles_unique_label.append(s)

                print('interpolation_mols valid {} / {}'
                      .format(len(valid_mols), len(interpolation_mols)))

                psize = (200, 200)
                with_seeds = [mol0] + valid_mols_unique + [mol1]
                # legends_with_seed = [smile0] + valid_mols_smiles_unique + [smile1]
                legends_with_seed = None
                mol_per_row = min(int((n_interpolation + 2) / 4), len(with_seeds))
                img = Draw.MolsToGridImage(with_seeds, legends=legends_with_seed,
                                           molsPerRow=mol_per_row,
                                           subImgSize=psize, useSVG=True)

                if not os.path.exists(f'{save_dir}/generated_interp'):
                    os.makedirs(f'{save_dir}/generated_interp')

                with open(
                        f'{save_dir}/generated_interp/{str(n).zfill(4)}_res_grid_valid{len(valid_mols)}_{len(interpolation_mols)}.svg',
                        'w') as f:
                    f.write(img)

                # img.save(
                #     f'{save_dir}/generated_interp/{str(n).zfill(4)}_res_grid_valid{len(valid_mols)}_{len(interpolation_mols)}.png')

            # if graphs
            if print_as_graph:
                graphs = val_dataset.get_full_graphs(data=list(zip(xm, adjm)),
                                                     attributed_node=val_dataset.is_attributed_node_dataset())

                if val_dataset.label_map is None:
                    inv_map = {i + 1: str(i) for i in range(x.shape[-1])}
                else:
                    inv_map = {v: k for k, v in val_dataset.label_map.items()}
                path = f'{save_dir}/generated_interp_graphs/{str(n).zfill(4)}'
                for i, graph in enumerate(graphs):
                    if graph is None:
                        continue
                    os.makedirs(path, exist_ok=True)
                    if val_dataset.is_attributed_node_dataset():
                        save_nx_graph_attr(graph, save_path=f'{path}/{str(i).zfill(4)}_{str(n).zfill(4)}')
                    else:
                        save_nx_graph(graph, inv_map, save_path=f'{path}/{str(i).zfill(4)}_{str(n).zfill(4)}',
                                      n_atom_type=x.shape[-1] - 1,
                                      colors=node_colors)
                # create grid
                nrow = math.ceil(math.sqrt(n_interpolation + 2))
                images = organise_data(path, nrow=nrow)
                # add borders
                margin = 2
                images[:, :-1, :margin, :] = 0
                images[:, :-1, -margin:, :] = 0
                images[:, :-1, :, :margin] = 0
                images[:, :-1, :, -margin:] = 0
                res_name = f'{str(n).zfill(4)}_res_grid'
                format(images, nrow=nrow, save_path=f'{save_dir}/generated_interp_graphs', res_name=res_name)


def create_figure_train_projections(model, train_dataset, std_noise, save_path, device, how_much=1, batch_size=20):
    size_pt_fig = 5

    loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    X_noise = []
    Z_noise = []
    Z = []
    tlabels = []
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(loader):
            inp, labels = data
            inp = train_dataset.format_data(inp, device)
            inp2 = inp.clone()
            if isinstance(inp, list):
                for i in range(len(inp)):
                    inp[i] = (inp[i] + torch.from_numpy(np.random.randn(*inp[i].shape)).float().to(device) * std_noise)
            else:
                inp = inp + torch.from_numpy(np.random.randn(*inp.shape)).float().to(device) * std_noise
            labels = labels.to(device)
            # inp = (data[0] + np.random.randn(*data[0].shape) * std_noise).float().to(device)
            # labels = data[1].to(device)
            _, _, _, out = model(inp, labels)
            X_noise.append(inp.detach().cpu().numpy())
            Z_noise.append(out.detach().cpu().numpy())

            _, _, _, out = model(inp2, labels)
            Z.append(out.detach().cpu().numpy())

            tlabels.append(labels.detach().cpu().numpy())
    tlabels = np.concatenate(tlabels, axis=0)
    X_noise = np.concatenate(X_noise, axis=0).reshape(len(train_dataset), -1)
    save_fig(X_noise, tlabels, size=size_pt_fig, save_path=f'{save_path}/X_space')

    Z_noise = np.concatenate(Z_noise, axis=0).reshape(len(train_dataset), -1)
    save_fig(Z_noise, tlabels, size=size_pt_fig, save_path=f'{save_path}/Z_space')

    Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)

    if not train_dataset.is_regression_dataset():
        label_zpcas = [None] * np.unique(tlabels).shape[0]
        # for each class learn PCA
        for j, label in enumerate(np.unique(tlabels)):
            Z_lab = Z_noise[np.where(tlabels == label)]

            # Learn the ZPCA projection
            label_zpcas[j] = (PCA(n_components=how_much), label)
            label_zpcas[j][0].fit(Z_lab)
        zpcas = label_zpcas
    else:
        # Learn the ZPCA projection
        zpca = PCA(n_components=how_much)
        zpca.fit(Z_noise)
        zpcas = [(zpca, 0)]

    np_z_noisy_all = []
    np_Z_all = []
    zpca_reconstruct_all = []
    X_lab_all = []
    X_ori_lab_all = []
    zpca_preimage_all = []
    np_proj_all = []
    gp_preimage_all = []
    labels_lab_all = []
    # By label
    for j, (zpca, label) in enumerate(zpcas):
        if not train_dataset.is_regression_dataset():
            idx = np.where(tlabels == label)
            X_lab = X_noise[idx]
            Z_lab = Z_noise[idx]
            X_ori_lab = train_dataset.X[idx]
            Z_ori_lab = Z[idx]
            labels_lab = tlabels[idx]
        else:
            X_lab = X_noise
            Z_lab = Z_noise
            X_ori_lab = train_dataset.X
            Z_ori_lab = Z
            labels_lab = tlabels

        labels_lab_all.append(labels_lab)

        np_z_noisy = Z_lab
        np_Z = Z_ori_lab

        # with ZPCA
        zpca_projection = zpca.transform(np_z_noisy)
        zpca_reconstruct = zpca.inverse_transform(zpca_projection)

        np_z_noisy_all.append(np_z_noisy)
        np_Z_all.append(np_Z)
        zpca_reconstruct_all.append(zpca_reconstruct)

        zpca_reconstruct = torch.from_numpy(zpca_reconstruct).type(torch.float).to(device)
        with torch.no_grad():
            zpca_preimage = model.reverse(zpca_reconstruct)

        zpca_preimage = zpca_preimage.detach().cpu().numpy()

        X_lab_all.append(X_lab)
        X_ori_lab_all.append(X_ori_lab)
        zpca_preimage_all.append(zpca_preimage)

        gmean = model.means[label].detach().cpu().numpy()
        # gp = model.gp[label][1:-1]
        # gp = (model.gp[label][1], np.identity(gmean.shape[0]))
        gp = (np.identity(gmean.shape[0]), model.gp[label][1])
        np_proj = project_inZ(Z_lab, params=(gmean, gp), how_much=how_much)
        proj = torch.from_numpy(np_proj).type(torch.float).to(device)

        np_proj_all.append(np_proj)

        with torch.no_grad():
            gp_preimage = model.reverse(proj)

        gp_preimage = gp_preimage.detach().cpu().numpy()

        gp_preimage_all.append(gp_preimage)

    label_max = np.max(train_dataset.true_labels)

    np_z_noisy_all = np.concatenate(np_z_noisy_all, axis=0)
    np_Z_all = np.concatenate(np_Z_all, axis=0)
    zpca_reconstruct_all = np.concatenate(zpca_reconstruct_all, axis=0)
    X_lab_all = np.concatenate(X_lab_all, axis=0)
    X_ori_lab_all = np.concatenate(X_ori_lab_all, axis=0)
    zpca_preimage_all = np.concatenate(zpca_preimage_all, axis=0)
    np_proj_all = np.concatenate(np_proj_all, axis=0)
    gp_preimage_all = np.concatenate(gp_preimage_all, axis=0)
    labels_lab_all = np.concatenate(labels_lab_all, axis=0)
    save_projection_fig(np_z_noisy_all, zpca_reconstruct_all, labels_lab_all, label_max=label_max, size=size_pt_fig,
                        save_path=f'{save_path}/noisytrain_projection_zpca_inZ')
    save_projection_fig(np_Z_all, zpca_reconstruct_all, labels_lab_all, label_max=label_max, size=size_pt_fig,
                        save_path=f'{save_path}/noisytrain_distance_zpca_inZ')

    save_projection_fig(X_lab_all, zpca_preimage_all, labels_lab_all, label_max=label_max, size=size_pt_fig,
                        save_path=f'{save_path}/noisytrain_projection_zpca')
    save_projection_fig(X_ori_lab_all, zpca_preimage_all, labels_lab_all, label_max=label_max, size=size_pt_fig,
                        save_path=f'{save_path}/noisytrain_distance_zpca')

    save_projection_fig(np_z_noisy_all, np_proj_all, labels_lab_all, label_max=label_max, size=size_pt_fig,
                        save_path=f'{save_path}/noisytrain_projection_gp_inZ')
    save_projection_fig(np_Z_all, np_proj_all, labels_lab_all, label_max=label_max, size=size_pt_fig,
                        save_path=f'{save_path}/noisytrain_distance_gp_inZ')

    save_projection_fig(X_lab_all, gp_preimage_all, labels_lab_all, label_max=label_max, size=size_pt_fig,
                        save_path=f'{save_path}/noisytrain_projection_gp')
    save_projection_fig(X_ori_lab_all, gp_preimage_all, labels_lab_all, label_max=label_max, size=size_pt_fig,
                        save_path=f'{save_path}/noisytrain_distance_gp')


def evaluate_graph_permutation(model, train_dataset, val_dataset, save_dir, device, batch_size=200):
    from utils.graphs.utils_datasets import batch_graph_permutation

    assert isinstance(train_dataset, GraphDataset), 'Evaluation for graph datasets !'
    train_dataset.permute_graphs_in_dataset()
    val_dataset.permute_graphs_in_dataset()

    create_folder(f'{save_dir}/eval_graph_perm')

    loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    # define an instance of the PairwiseDistance
    pdist = torch.nn.PairwiseDistance(p=2)

    # n_permutation test
    n_perm = 10

    model.eval()
    shuffle_ids = []
    outs = []
    perm_dists = []
    with torch.no_grad():
        for j, data in enumerate(loader):
            inp, labels = data
            inp = train_dataset.format_data(inp, device)
            labels = labels.to(device)
            log_p, distloss, logdet, out = model(inp, labels)
            outs.append(out.detach().cpu())

            for n in range(n_perm):
                inp_p, sh_id = batch_graph_permutation(inp, return_sh_id=True)
                while sh_id in shuffle_ids:
                    inp_p, sh_id = batch_graph_permutation(inp, return_sh_id=True)
                shuffle_ids.append(sh_id)

                _, _, _, out2 = model(inp_p, labels)
                outs.append(out2.detach().cpu())
                perm_dists.append(pdist(out, out2).detach().cpu().unsqueeze(0))

                D = torch.cdist(out, out2)
                for i in range(D.shape[0]):
                    sorted_id_row = torch.argsort(D[i])
                    first_sorted_id = sorted_id_row[:10]
                    cpt = 0
                    while i not in first_sorted_id:
                        cpt += 1
                        first_sorted_id = sorted_id_row[:(10 + cpt)]
                    graphs = tuple([v[first_sorted_id].detach().cpu() for v in inp_p])

                    ori_mol = valid_mol(construct_mol(inp[0][i].detach().cpu(), inp[1][i].detach().cpu(),
                                                      train_dataset.atomic_num_list))
                    mols = [valid_mol(construct_mol(x_elem, adj_elem, train_dataset.atomic_num_list))
                            for x_elem, adj_elem in zip(*graphs)]
                    mols = [ori_mol] + mols

                    legends = ['ori'] + [f'{d:.4f}' for d in list(D[i][first_sorted_id].detach().cpu().numpy())]

                    psize = (200, 200)
                    img = Draw.MolsToGridImage(mols, legends=legends,
                                               molsPerRow=math.ceil(math.sqrt(D.shape[0])),
                                               subImgSize=psize,
                                               # useSVG=True
                                               )
                    img.save(f'{save_dir}/eval_graph_perm/{j:04}_{n:04}_{i:04}.png')

            perm_dists = torch.cat(perm_dists, 0)
            maxs = torch.max(perm_dists, 0)
            mins = torch.min(perm_dists, 0)
            means = torch.mean(perm_dists, 0)

            print(f'{maxs}, {mins}, {means}.')
            break


def launch_evaluation(dataset_name, model, gaussian_params, train_dataset, val_dataset, save_dir, device, batch_size):
    dim_per_label, n_dim = train_dataset.get_dim_per_label(return_total_dim=True)

    # generate flows
    # std_noise = .1 if dataset_name in ['double_moon', 'single_moon', 'swissroll'] else None
    # # fig_limits = ((-23,23),(-4,4))
    # fig_limits = None
    # model.interpret_transformation(train_dataset, save_dir, device, std_noise=std_noise, fig_limits=fig_limits)

    eval_type = args.eval_type
    if eval_type == 'classification':
        dataset_name_eval = ['mnist', 'cifar10', 'double_moon', 'iris', 'bcancer'] + GRAPH_CLASSIFICATION_DATASETS
        assert dataset_name in dataset_name_eval, f'Classification can only be evaluated on {dataset_name_eval}'
        # predmodel = evaluate_classification(model, train_dataset, val_dataset, save_dir, device, batch_size=batch_size)
        evaluate_classification(model, train_dataset, val_dataset, save_dir, device, batch_size=batch_size)
        # _, Z = create_figures_XZ(model, train_dataset, save_dir, device, std_noise=0.1,
        #                          only_Z=isinstance(train_dataset, GraphDataset), batch_size=batch_size)
        # print_as_mol = True
        # print_as_graph = False
        # refresh_means = False
        # print(f'(print_as_mol, print_as_graph, refresh_means) are set manually to '
        #       f'({print_as_mol},{print_as_graph},{refresh_means}).')
        # computed_means = model.refresh_classification_mean_classes(Z, train_dataset.true_labels) if refresh_means \
        #     else None
        # evaluate_preimage(model, val_dataset, device, save_dir, print_as_mol=print_as_mol,
        #                   print_as_graph=print_as_graph, eval_type=eval_type, means=computed_means, batch_size=batch_size)
        # evaluate_preimage2(model, val_dataset, device, save_dir, n_y=20, n_samples_by_y=12, print_as_mol=print_as_mol,
        #                    print_as_graph=print_as_graph, eval_type=eval_type, predmodel=predmodel,
        #                    means=computed_means, debug=True, batch_size=batch_size)
        # if isinstance(val_dataset, GraphDataset):
        #     evaluate_graph_interpolations(model, val_dataset, device, save_dir, n_sample=9, n_interpolation=30, Z=Z,
        #                                   print_as_mol=print_as_mol, print_as_graph=print_as_graph, eval_type=eval_type,
        #                                   label=None)
    elif eval_type == 'generation':
        dataset_name_eval = ['mnist', 'cifar10', 'olivetti_faces']
        assert dataset_name in dataset_name_eval, f'Generation can only be evaluated on {dataset_name_eval}'
        # GENERATION
        create_folder(f'{save_dir}/test_generation')

        img_size = train_dataset.in_size
        z_shape = model.calc_last_z_shape(img_size)
        from utils.training import AddGaussianNoise
        from torchvision import transforms
        val_dataset.transform = transforms.Compose(val_dataset.transform.transforms + [AddGaussianNoise(0., .2)])
        print('Gaussian noise added to transforms...')
        debug = True
        # transformation_interpretation(model, val_dataset, device, save_dir, debug=False)
        evaluate_image_interpolations(model, val_dataset, device, save_dir, n_sample=6, n_interpolation=10, label=None,
                                      debug=debug)

        # how_much = [1, 10, 30, 50, 78]
        # how_much = [dim_per_label, n_dim]
        # how_much = [1, dim_per_label]
        how_much = list(np.linspace(1, dim_per_label, 6, dtype=np.int))
        print(f'how_much is set manually to {how_much}.')
        for n in how_much:
            test_generation_on_eigvec(model, val_dataset, gaussian_params=gaussian_params, z_shape=z_shape,
                                      how_much_dim=n, device=device, sample_per_label=49, save_dir=save_dir,
                                      debug=debug)
        generate_meanclasses(model, train_dataset, device, save_dir, debug=debug, batch_size=batch_size)
    elif eval_type == 'projection':
        img_size = train_dataset.in_size
        z_shape = model.calc_last_z_shape(img_size)
        # PROJECTIONS
        proj_type = 'gp'
        # batch_size = 20  # 100
        eval_gaussian_std = .2  # .1
        print(f'(proj_type, batch_size, eval_gaussian_std) are set manually to '
              f'({proj_type},{batch_size},{eval_gaussian_std}).')
        if dataset_name in ['mnist', 'cifar10', 'olivetti_faces']:
            noise_types = ['gaussian', 'speckle', 'poisson', 's&p']
            how_much = list(np.linspace(1, dim_per_label, 6, dtype=np.int))
            # how_much = list(np.linspace(1, dim_per_label, 6, dtype=np.int)) + list(
            #     np.linspace(int(dim_per_label + dim_per_label / 6),
            #                 int(dim_per_label + dim_per_label / 6) + dim_per_label, 6, dtype=np.int))
            # how_much = [1,
            #             np.min(np.histogram(train_dataset.true_labels, bins=np.unique(train_dataset.true_labels))[0])]
            print(f'how_much is set manually to {how_much}.')
            for noise_type in noise_types:
                compression_evaluation(model, train_dataset, val_dataset, gaussian_params=gaussian_params,
                                       z_shape=z_shape, how_much=how_much, device=device, save_dir=save_dir,
                                       proj_type=proj_type, noise_type=noise_type, eval_gaussian_std=eval_gaussian_std,
                                       batch_size=batch_size)
                evaluate_projection_1model(model, train_dataset, val_dataset, gaussian_params=gaussian_params,
                                           z_shape=z_shape, how_much=dim_per_label, device=device, save_dir=save_dir,
                                           proj_type=proj_type, noise_type=noise_type,
                                           eval_gaussian_std=eval_gaussian_std,
                                           batch_size=batch_size)
        else:
            # elif dataset_name in ['single_moon', 'double_moon']:
            noise_type = 'gaussian'
            std_noise = .1
            n_principal_dim = np.count_nonzero(gaussian_params[0][-2] > 1)
            create_figure_train_projections(model, train_dataset, std_noise=std_noise, save_path=save_dir,
                                            device=device, how_much=n_principal_dim, batch_size=batch_size)
            # evaluate distance n times to calculate the p-value
            n_times = 20
            kpca_types = ['linear', 'rbf', 'poly', 'sigmoid']
            proj_type = 'zpca_l'
            print(f'(n_times, kpca_types, proj_type) are set manually to '
                  f'({n_times},{kpca_types},{proj_type}).')
            distance_results = {ktype + '-PCA': [] for ktype in kpca_types}
            distance_results['Our-' + proj_type] = []
            distance_results['NoiseDist'] = []
            for n in range(n_times):
                res = evaluate_distances(model, train_dataset, val_dataset, gaussian_params=gaussian_params,
                                         z_shape=z_shape, how_much=n_principal_dim,
                                         kpca_types=kpca_types,
                                         device=device, save_dir=save_dir, proj_type=proj_type, noise_type=noise_type,
                                         eval_gaussian_std=std_noise, batch_size=batch_size)
                for ktype in kpca_types:
                    distance_results[ktype + '-PCA'].append(res[ktype + '-PCA'])
                distance_results['Our-' + proj_type].append(res['Our-' + proj_type])
                distance_results['NoiseDist'].append(res['NoiseDist'])
            print(distance_results)
            # p-value
            mean_noisedist = np.mean(distance_results['NoiseDist'])
            print('Mean noise dist: ' + str(mean_noisedist))
            mean_score = np.mean(distance_results['Our-' + proj_type])
            print('Mean score: ' + str(mean_score))
            res_pvalue = evaluate_p_value(distance_results)
            if res_pvalue is not None:
                H, p = res_pvalue
                score_str = 'Kruskal-Wallis H-test, H: ' + str(H) + ', p-value: ' + str(p)
                print(score_str)

                # by pairs
                res_pvalue = evaluate_p_value(distance_results, by_pairs=True)
                for k, v in res_pvalue.items():
                    H, p = v
                    score_str = 'Kruskal-Wallis H-test with ' + str(k) + ', H: ' + str(H) + ', p-value: ' + str(p)
                    print(score_str)
        # else:
        #     dataset_name_eval = ['mnist', 'single_moon', 'double_moon']
        #     assert dataset_name in dataset_name_eval, f'Projection can only be evaluated on {dataset_name_eval}'
    elif eval_type == 'regression':
        assert train_dataset.is_regression_dataset(), 'the dataset is not made for regression purposes'
        predmodel = evaluate_regression(model, train_dataset, val_dataset, save_dir, device, batch_size=batch_size)
        _, Z = create_figures_XZ(model, train_dataset, save_dir, device, std_noise=0.1,
                                 only_Z=isinstance(train_dataset, GraphDataset), batch_size=batch_size)
        print_as_mol = True
        print_as_graph = False
        print(f'(print_as_mol, print_as_graph) are set manually to '
              f'({print_as_mol},{print_as_graph}).')
        evaluate_preimage(model, val_dataset, device, save_dir, print_as_mol=print_as_mol,
                          print_as_graph=print_as_graph, batch_size=batch_size)
        evaluate_preimage2(model, val_dataset, device, save_dir, n_y=12, n_samples_by_y=1,
                           print_as_mol=print_as_mol, print_as_graph=print_as_graph, predmodel=predmodel, debug=True,
                           batch_size=batch_size)
        if isinstance(val_dataset, GraphDataset):
            evaluate_graph_interpolations(model, val_dataset, device, save_dir, n_sample=100, n_interpolation=30, Z=Z,
                                          print_as_mol=print_as_mol, print_as_graph=print_as_graph)


def main(args):
    print(args)
    if args.seed is not None:
        set_seed(args.seed)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]

    # DATASET #
    dataset = load_dataset(args, dataset_name, model_type, to_evaluate=True, add_feature=args.add_feature)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_path = f'{args.folder}/save/best_{args.model_to_use}_model.pth'
    assert os.path.exists(flow_path), f'snapshot path {flow_path} does not exists'
    folder_name = args.folder.split('/')[-1]

    # Retrieve parameters from name
    n_block, n_flow, mean_of_eigval, dim_per_label, label, fixed_eigval, uniform_eigval, gaussian_eigval, \
    reg_use_var, split_graph_dim = retrieve_params_from_name(folder_name)

    if label is not None:
        dataset.reduce_dataset('one_class', label=label)

    # Load the splits of the dataset used in the training phase
    train_idx_path = f'{args.folder}/train_idx.npy'
    val_idx_path = f'{args.folder}/val_idx.npy'
    train_dataset, val_dataset = load_split_dataset(dataset, train_idx_path, val_idx_path,
                                                    reselect_val_idx=args.reselect_val_idx)

    # reduce train dataset size (fitting too long)
    # print('Train dataset reduced in order to accelerate. (stratified)')
    # train_dataset.reduce_dataset_ratio(0.05, stratified=True)

    dim_per_label, n_dim = dataset.get_dim_per_label(return_total_dim=True)

    # initialize gaussian params
    gaussian_params = initialize_gaussian_params(args, dataset, fixed_eigval, uniform_eigval, gaussian_eigval,
                                                 mean_of_eigval, dim_per_label, 'isotrope' in folder_name,
                                                 split_graph_dim)

    if model_type == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        args_cglow.n_flow = n_flow
        n_channel = dataset.n_channel
        model_single = load_cglow_model(args_cglow, n_channel, gaussian_params=gaussian_params,
                                        learn_mean='lmean' in folder_name, device=device)
    elif model_type == 'seqflow':
        model_single = load_seqflow_model(n_dim, n_flow, gaussian_params=gaussian_params,
                                          learn_mean='lmean' in folder_name, reg_use_var=reg_use_var, dataset=dataset)

    elif model_type == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        args_ffjord.n_block = n_block
        model_single = load_ffjord_model(args_ffjord, n_dim, gaussian_params=gaussian_params,
                                         learn_mean='lmean' in folder_name, reg_use_var=reg_use_var, dataset=dataset)
    elif model_type == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        model_single = load_moflow_model(args_moflow,
                                         gaussian_params=gaussian_params,
                                         learn_mean='lmean' in folder_name, reg_use_var=reg_use_var,
                                         dataset=dataset)
    else:
        assert False, 'unknown model type!'
    model_single = WrappedModel(model_single)
    model_single.load_state_dict(torch.load(flow_path, map_location=device))
    # Convert old FFJORD model params
    # params = torch.load(flow_path, map_location=device)
    # model_parms = params['state_dict']
    # n_params = {}
    # for k,v in model_parms.items():
    #     st = 'module.model.' + k
    #     n_params[st] = v
    # args_l = params['args']
    # n_params['module.means'] = params['means'] # n_params['module.means'] = torch.zeros(1,2).to(device)
    # model_single.load_state_dict(n_params)
    # torch.save(model_single.state_dict(), flow_path)

    # Convert old Glow model params
    # params = torch.load(flow_path, map_location=device)
    # model_params = ['module.blocks', 'module.n_channel', 'module.n_block']
    # n_params = {}
    # for k, v in params.items():
    #     if '.'.join(k.split('.')[:2]) in model_params:
    #         st = 'module.model.' + '.'.join(k.split('.')[1:])
    #         n_params[st] = v
    #     else:
    #         n_params[k] = v
    # model_single.load_state_dict(n_params)
    model = model_single.module
    model = model.to(device)
    model.eval()

    os.chdir(args.folder)

    save_dir = './save'
    create_folder(save_dir)

    # evaluate_graph_permutation(model, train_dataset, val_dataset, save_dir, device, batch_size=batch_size)
    batch_size = args.batch_size
    launch_evaluation(dataset_name, model, gaussian_params, train_dataset, val_dataset, save_dir, device, batch_size)


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
    parser.add_argument("--add_feature", type=int, default=None)
    args = parser.parse_args()
    args.seed = 2
    main(args)
