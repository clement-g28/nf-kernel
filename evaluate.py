import numpy as np
from PIL import Image, ImageDraw
import time
import math
import os
import re
import torch
from torchvision import utils

from utils.utils import set_seed, create_folder, save_every_pic

from utils.custom_glow import CGlow, WrappedModel, initialize_gaussian_params
from utils.toy_models import load_seqflow_model, load_ffjord_model
from utils.dataset import ImDataset, SimpleDataset
from utils.density import construct_covariance
from utils.testing import learn_or_load_modelhyperparams, generate_sample, project_inZ, testing_arguments, noise_data
from utils.training import ffjord_arguments

from sklearn.decomposition import PCA, KernelPCA

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def test_generation_on_eigvec(model_single, gaussian_params, z_shape, how_much_dim, device, sample_per_label=10,
                              save_dir='./save'):
    create_folder(f'{save_dir}/test_generation')

    all_generation = []
    n_image_per_lab = 10
    for i, gaussian_param in enumerate(gaussian_params):
        mean = model_single.means[i].detach().cpu().numpy().squeeze()
        eigenvec = gaussian_param[1]
        eigenval = gaussian_param[2]
        indexes = np.flip(np.argsort(eigenval))
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

        images = model_single.reverse(z_sample).cpu().data
        images = val_dataset.rescale(images)
        utils.save_image(
            images,
            f"{save_dir}/test_generation/label{str(i)}_{how_much_dim}dim.png",
            normalize=True,
            nrow=10,
            range=(0, 255),
        )
        all_generation.append(images[:n_image_per_lab])

    mean_images = generate_meanclasses(model_single, val_dataset, device)
    all_with_means = []
    for i, n_generation in enumerate(all_generation):
        all_with_means.append(n_generation)
        all_with_means.append(np.expand_dims(mean_images[i], axis=0))
    all_generation = np.concatenate(all_generation, axis=0)

    utils.save_image(
        torch.from_numpy(all_generation),
        f"{save_dir}/test_generation/all_label_{how_much_dim}dim.png",
        normalize=True,
        nrow=10,
        range=(0, 255),
    )

    all_with_means = np.concatenate(all_with_means, axis=0)

    methods = ([str(i) for i in range(0, n_image_per_lab)] + ['mean']) * len(gaussian_params)
    labels = [[g[-1]] * (n_image_per_lab + 1) for g in gaussian_params]
    labels = [item for sublist in labels for item in sublist]
    save_every_pic(f'{save_dir}/test_generation/every_pics/{how_much_dim}', all_with_means, methods, labels)


def evaluate_projection_1model(model, train_dataset, val_dataset, gaussian_params, z_shape, how_much, device,
                               save_dir='./save', proj_type='gp', noise_type='gaussian', eval_gaussian_std=.2):
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
                                                       eval_gaussian_std=eval_gaussian_std)

    grid_im = np.concatenate(grid_im, axis=0)
    grid_im = val_dataset.rescale(grid_im)

    used_method = ['ori', 'noisy', 'linear'] + kpca_types + [proj_type]
    methods = used_method * len(gaussian_params)
    labels = [[g[-1]] * len(used_method) for g in gaussian_params]
    labels = [item for sublist in labels for item in sublist]
    save_every_pic(f'{save_dir}/projections/{noise_type}/every_pics/{proj_type}', grid_im, methods, labels,
                   add_str=proj_type)

    nrow = math.floor(grid_im.shape[0] / np.unique(val_dataset.true_labels).shape[0])
    utils.save_image(
        torch.from_numpy(grid_im),
        f"{save_dir}/projections/{noise_type}/projection_eval_{proj_type}.png",
        normalize=True,
        nrow=nrow,
        range=(0, 255),
    )


def projection_evaluation(model, train_dataset, val_dataset, gaussian_params, z_shape, how_much, kpca_types, device,
                          save_dir='./save', proj_type='gp', noise_type='gaussian', eval_gaussian_std=.2):
    create_folder(f'{save_dir}/projections')

    train_dataset.ori_X = train_dataset.X
    val_dataset.ori_X = val_dataset.X
    train_dataset.ori_true_labels = train_dataset.true_labels
    val_dataset.ori_true_labels = val_dataset.true_labels
    train_dataset.idx = np.array([i for i in range(train_dataset.X.shape[0])])
    val_dataset.idx = np.array([i for i in range(val_dataset.X.shape[0])])

    train_noised = noise_data(train_dataset.X / 255, noise_type=noise_type, gaussian_mean=0,
                              gaussian_std=eval_gaussian_std)
    train_dataset_noised = train_dataset.duplicate()
    train_dataset_noised.X = (train_noised * 255).astype(np.uint8)
    train_normalized_noised = (train_noised - train_dataset.norm_mean) / train_dataset.norm_std

    batch_size = 20
    loader = train_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    if proj_type in ['zpca', 'zpca_l']:
        Z = []
        tlabels = []
        with torch.no_grad():
            for j, data in enumerate(loader):
                inp = data[0].float().to(device)
                labels = data[1].to(torch.int64).to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                Z.append(out.detach().cpu().numpy())
                tlabels.append(labels.detach().cpu().numpy())
        Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)
        tlabels = np.concatenate(tlabels, axis=0)

    val_noised = noise_data(val_dataset.X / 255, noise_type=noise_type, gaussian_mean=0,
                            gaussian_std=eval_gaussian_std)
    val_dataset_noised = val_dataset.duplicate()
    val_dataset_noised.X = (val_noised * 255).astype(np.uint8)
    val_normalized_noised = (val_noised - train_dataset.norm_mean) / train_dataset.norm_std

    val_loader = val_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

    val_inZ = []
    elabels = []
    with torch.no_grad():
        for j, data in enumerate(val_loader):
            inp = data[0].float().to(device)
            labels = data[1].to(torch.int64).to(device)
            log_p, distloss, logdet, out = model(inp, labels)
            val_inZ.append(out.detach().cpu().numpy())
            elabels.append(labels.detach().cpu().numpy())
    val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(val_dataset), -1)
    elabels = np.concatenate(elabels, axis=0)

    val_normalized = (val_dataset.X / 255 - val_dataset.norm_mean) / val_dataset.norm_std

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

        images = model.reverse(z_b)
        all_im.append(images.detach().cpu().numpy())
    all_im = np.concatenate(all_im, axis=0)

    our_dist = np.sum(np.abs(all_im.reshape(all_im.shape[0], -1) - ordered_val.reshape(ordered_val.shape[0], -1)),
                      axis=1)
    mean_dist = np.mean(our_dist)
    print(f'Our approach {proj_type} : {mean_dist}')
    distances_results[f'Our-{proj_type}'] = mean_dist

    # Save one of each
    grid_im = []
    vis_index = 10
    for i, gaussian_param in enumerate(gaussian_params):
        label = gaussian_param[-1]
        ind = np.where(ordered_elabels == label)[0]
        grid_im.append(np.expand_dims(val_normalized[np.where(val_dataset.true_labels == label)][vis_index], axis=0))
        grid_im.append(
            np.expand_dims(val_normalized_noised[np.where(val_dataset.true_labels == label)][vis_index], axis=0))
        # grid_im.append(np.expand_dims(pca_projs[ind][vis_index].reshape(val_dataset.X[0].shape), axis=0))
        for kpca_proj in kpca_projs:
            grid_im.append(np.expand_dims(kpca_proj[ind][vis_index].reshape(val_dataset.X[0].shape), axis=0))
        grid_im.append(np.expand_dims(all_im[ind][vis_index], axis=0))

    return grid_im, distances_results


def evaluate_classification(model, train_dataset, val_dataset, save_dir, device, fithyperparam=True):
    # Compute results with our approach if not None
    if model is not None:
        batch_size = 20
        loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

        start = time.time()
        Z = []
        tlabels = []
        model.eval()
        with torch.no_grad():
            for j, data in enumerate(loader):
                inp = data[0].float().to(device)
                labels = data[1].to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                Z.append(out.detach().cpu().numpy())
                tlabels.append(labels.detach().cpu().numpy())
        Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)
        tlabels = np.concatenate(tlabels, axis=0)
        end = time.time()
        print(f"time to get Z_train from X_train: {str(end - start)}, batch size: {batch_size}")

        # Learn SVC
        start = time.time()
        zlinsvc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        zlinsvc.fit(Z, tlabels)
        print(zlinsvc)
        end = time.time()
        print(f"time to fit linear svc in Z : {str(end - start)}")

        val_loader = val_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

        start = time.time()
        val_inZ = []
        elabels = []
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                inp = data[0].float().to(device)
                labels = data[1].to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                val_inZ.append(out.detach().cpu().numpy())
                elabels.append(labels.detach().cpu().numpy())
        val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(val_dataset), -1)
        elabels = np.concatenate(elabels, axis=0)
        end = time.time()
        print(f"time to get Z_val from X_val : {str(end - start)}")

        zsvc_score = zlinsvc.score(val_inZ, elabels)
        print(f'Our approach : {zsvc_score}')

        # Misclassified data
        predictions = zlinsvc.predict(val_inZ)
        misclassif_i = np.where((predictions == elabels) == False)
        if misclassif_i[0].shape[0] > 0:
            z_sample = torch.from_numpy(val_inZ[misclassif_i].reshape(misclassif_i[0].shape[0], *z_shape)).float().to(
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

    X_train = train_dataset.X.reshape(train_dataset.X.shape[0], -1)
    labels_train = train_dataset.true_labels

    start = time.time()
    kernel_name = 'linear'
    if fithyperparam:
        param_gridlin = [{'SVC__kernel': [kernel_name], 'SVC__C': np.logspace(-5, 3, 10)}]
        linsvc = learn_or_load_modelhyperparams(X_train, labels_train, kernel_name, param_gridlin, save_dir,
                                                model_type=('SVC', SVC()), scaler=True)
    else:
        linsvc = make_pipeline(StandardScaler(), SVC(kernel=kernel_name))
        linsvc.fit(X_train, labels_train)
        print(f'Fitting done.')
    end = time.time()
    print(f"time to fit linear svc : {str(end - start)}")

    # krr_types = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    ksvc_types = ['rbf', 'poly', 'sigmoid']
    ksvc_params = [
        {'SVC__kernel': ['rbf'], 'SVC__gamma': np.logspace(-5, 3, 10), 'SVC__C': np.logspace(-5, 3, 10)},
        {'SVC__kernel': ['poly'], 'SVC__gamma': np.logspace(-5, 3, 5), 'SVC__degree': np.linspace(1, 2, 2),
         'SVC__C': np.logspace(-5, -3, 2)},
        {'SVC__kernel': ['sigmoid'], 'SVC__C': np.logspace(-5, 3, 10)}
    ]

    ksvcs = [None] * len(ksvc_types)
    for i, ksvc_type in enumerate(ksvc_types):
        start = time.time()
        if fithyperparam:
            ksvcs[i] = learn_or_load_modelhyperparams(X_train, labels_train, ksvc_type, [ksvc_params[i]], save_dir,
                                                      model_type=('SVC', SVC()), scaler=True)
        else:
            ksvcs[i] = make_pipeline(StandardScaler(), SVC(kernel=ksvc_type))
            ksvcs[i].fit(X_train, labels_train)
            print(f'Fitting done.')
        end = time.time()
        print(f"time to fit {ksvc_type} svc : {str(end - start)}")

    X_val = val_dataset.X.reshape(val_dataset.X.shape[0], -1)
    labels_val = val_dataset.true_labels

    ksvc_scores = []
    for ksvc in ksvcs:
        ksvc_scores.append([])

    svc_score = linsvc.score(X_val, labels_val)

    for i, ksvc in enumerate(ksvcs):
        ksvc_score = ksvc.score(X_val, labels_val)
        ksvc_scores[i].append(ksvc_score)

    print('Predictions scores :')
    print(f'SVC : {np.mean(svc_score)}')
    for j, kpca_type in enumerate(ksvc_types):
        print(f'KernelSVC ({kpca_type}) : {np.mean(ksvc_scores[j])}')
    print(f'Our approach : {zsvc_score}')


def generate_meanclasses(model, dataset, device):
    model.eval()
    create_folder(f'{save_dir}/test_generation')

    batch_size = 20
    loader = dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

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
                       save_dir='./save', proj_type='gp', noise_type='gaussian', eval_gaussian_std=.1):
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

    batch_size = 20
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

    # from get_best_projection import save_fig
    # save_fig(ordered_val, all_res, ordered_elabels, label_max=1, size=5,
    #          save_path=f'{save_dir}/projection_zpca_model')

    return distances_results


if __name__ == "__main__":
    choices = ['classification', 'projection', 'generation']
    best_model_choices = ['classification', 'projection']
    parser = testing_arguments()
    parser.add_argument('--eval_type', type=str, default='classification', choices=choices, help='evaluation type')
    parser.add_argument('--model_to_use', type=str, default='classification', choices=best_model_choices,
                        help='what best model to use for the evaluation')
    args = parser.parse_args()
    print(args)
    set_seed(0)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]
    # DATASET #
    if dataset_name == 'mnist':
        dataset = ImDataset(dataset_name=dataset_name)
        n_channel = dataset.n_channel
    else:
        dataset = SimpleDataset(dataset_name=dataset_name)
        n_channel = 1

    img_size = dataset.im_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_path = f'{args.folder}/save/best_{args.model_to_use}_model.pth'
    assert os.path.exists(flow_path), f'snapshot path {flow_path} does not exists'
    folder_name = args.folder.split('/')[-1]

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
            dataset.reduce_dataset('one_class', label=label)
        elif 'manualeigval' in split:
            manualeigval_split = split
            fixed_eigval = list(map(float, manualeigval_split.replace("manualeigval", "").split('-')))
            print(f'Flow trained with fixed eigenvalues: {fixed_eigval}')
        elif 'eigvaluniform' in split:
            uniform_eigval = True
            uniform_split = split
            mean_of_eigval = uniform_split.replace("eigvaluniform", "")
            mean_of_eigval = float(mean_of_eigval.replace("-", "."))
            print(f'Flow trained with uniform eigenvalues: {mean_of_eigval}')
        elif 'eigvalgaussian' in split:
            gaussian_split = split
            in_split = gaussian_split.split("std")
            mean_of_eigval = int(in_split[0].replace("eigvalgaussian", ""))
            std_value = float(str(in_split[-1]).replace('-', '.'))
            gaussian_eigval = [0.0, std_value]
            print(f'Flow trained with gaussian eigenvalues params: {mean_of_eigval},{gaussian_eigval}')
        elif 'manualeigval' in split:
            manual_split = split
            mean_of_eigval = int(in_split[0].replace("manualeigval", ""))
            std_value = float(str(in_split[-1]).replace('-', '.'))
            gaussian_eigval = [0.0, std_value]
            print(f'Flow trained with gaussian eigenvalues params: {mean_of_eigval},{gaussian_eigval}')
        elif 'dimperlab' in split:
            dpl_split = split
            dim_per_label = int(dpl_split.replace("dimperlab", ""))
            print(f'Flow trained with dimperlab: {dim_per_label}')

    # Model params
    affine = args.affine
    no_lu = args.no_lu

    # Load the splits of the dataset used in the training phase
    train_idx_path = f'{args.folder}/train_idx.npy'
    val_idx_path = f'{args.folder}/val_idx.npy'
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
        val_dataset = ImDataset(dataset_name=dataset_name, test=True)

    n_dim = dataset.X[0].shape[0]
    for sh in dataset.X[0].shape[1:]:
        n_dim *= sh
    if not dim_per_label:
        uni = np.unique(dataset.true_labels)
        dim_per_label = math.floor(n_dim / len(uni))

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
    gaussian_params = initialize_gaussian_params(dataset, eigval_list, isotrope='isotrope' in folder_name,
                                                 dim_per_label=dim_per_label, fixed_eigval=fixed_eigval)
    if model_type == 'cglow':
        # Load model
        model_single = CGlow(n_channel, n_flow, n_block, affine=affine, conv_lu=not no_lu,
                             gaussian_params=gaussian_params, device=device, learn_mean='lmean' in folder_name)
    elif model_type == 'seqflow':
        model_single = load_seqflow_model(n_dim, n_flow, gaussian_params=gaussian_params,
                                          learn_mean='lmean' in folder_name)

    elif model_type == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        args_ffjord.n_block = n_block
        model_single = load_ffjord_model(args_ffjord, n_dim, gaussian_params=gaussian_params,
                                         learn_mean='lmean' in folder_name)

    else:
        assert False, 'unknown model type!'
    z_shape = model_single.calc_last_z_shape(img_size)
    model_single = WrappedModel(model_single)
    model_single.load_state_dict(torch.load(flow_path, map_location=device))
    model = model_single.module
    model = model.to(device)
    model.eval()

    os.chdir(args.folder)

    save_dir = './save'
    create_folder(save_dir)

    eval_type = args.eval_type
    if eval_type == 'classification':
        dataset_name_eval = ['mnist', 'double_moon', 'iris', 'bcancer']
        assert dataset_name in dataset_name_eval, f'Classification can only be evaluated on {dataset_name_eval}'
        evaluate_classification(model, train_dataset, val_dataset, save_dir, device)
    elif eval_type == 'generation':
        dataset_name_eval = ['mnist']
        assert dataset_name in dataset_name_eval, f'Generation can only be evaluated on {dataset_name_eval}'
        # GENERATION
        how_much = [1, 10, 30, 50, 78]
        for n in how_much:
            test_generation_on_eigvec(model, gaussian_params=gaussian_params, z_shape=z_shape, how_much_dim=n,
                                      device=device, sample_per_label=10, save_dir=save_dir)
        generate_meanclasses(model, train_dataset, device)
    elif eval_type == 'projection':
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
