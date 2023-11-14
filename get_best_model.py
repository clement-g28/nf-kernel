import numpy as np
import math
import torch
import glob
import os

from utils.custom_glow import WrappedModel
from utils.dataset import GraphDataset

from utils.utils import set_seed, create_folder, save_fig, load_dataset

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from utils.dataset import ImDataset, SimpleDataset

from utils.testing import testing_arguments, noise_data, save_modelhyperparams, retrieve_params_from_name, \
    learn_or_load_modelhyperparams, initialize_gaussian_params, prepare_model_loading_params, load_split_dataset, \
    load_model_from_params, project_inZ


def evaluate_classification(t_model_params, train_dataset, eval_dataset, full_dataset, save_dir, device,
                            with_train=False, cus_load_func=None, batch_size=200):
    svc_scores = []

    best_score = 0
    best_i = 0
    best_score_train = 0
    best_i_train = 0
    best_score_str_train = ''
    best_score_str = ''

    if isinstance(train_dataset, GraphDataset):  # Permutations for Graphs
        train_dataset.permute_graphs_in_dataset()
        eval_dataset.permute_graphs_in_dataset()

    start_from = None
    # start_from = 454
    for i, model_loading_params in enumerate(t_model_params):
        if start_from is not None and i < start_from:
            continue
        model_single = load_model_from_params(model_loading_params, full_dataset)
        try:
            if not cus_load_func:
                # model_single = WrappedModel(model_single)
                model = model_single
                model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
                # model = model_single.module
            else:
                model = cus_load_func(model_single, model_loading_params['loading_path'])
        except:
            print(f'Exception while loading model {i}.')
            continue
        model = model.to(device)
        model.eval()

        loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

        print('Computing Z...')
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

        print('Fit SVC on Z...')
        # Learn SVC
        # param_gridlin = [{'SVC__kernel': ['linear'], 'SVC__C': np.array([1])}]
        param_gridlin = [{'SVC__kernel': ['linear'], 'SVC__C': np.concatenate((np.logspace(-5, 2, 10), np.array([1])))}]
        model_type = ('SVC', SVC())
        scaler = False
        svc = learn_or_load_modelhyperparams(Z, tlabels, 'zlinear', param_gridlin, save_dir,
                                             model_type=model_type, scaler=scaler, save=False,
                                             force_train=True)
        # svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        # svc.fit(Z, tlabels)

        print('Evaluate...')
        if isinstance(train_dataset, GraphDataset):  # Permutations for Graphs
            n_permutation = 10
            scores = []
            scores_train = []
            for n_perm in range(n_permutation):
                eval_dataset.permute_graphs_in_dataset()

                val_loader = eval_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

                val_inZ = []
                elabels = []
                with torch.no_grad():
                    for j, data in enumerate(val_loader):
                        inp, labels = data

                        inp = eval_dataset.format_data(inp, device)
                        labels = labels.to(device)
                        log_p, distloss, logdet, out = model(inp, labels)
                        val_inZ.append(out.detach().cpu().numpy())
                        elabels.append(labels.detach().cpu().numpy())
                val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(eval_dataset), -1)
                elabels = np.concatenate(elabels, axis=0)

                svc_score = svc.score(val_inZ, elabels)
                svc_scores.append(svc_score)

                svc_score_train = svc.score(Z, tlabels)

                scores.append(svc_score)
                scores_train.append(svc_score_train)
            mean_score = np.mean(scores)
            mean_score_train = np.mean(scores_train)
            score_str = f'Our approach ({i}) : {scores}, (train score : {scores_train}) \n' \
                        f'Mean Scores: {mean_score}, (train score : {mean_score_train})'
            print(score_str)
            # if zridge_score >= best_score:
            if mean_score_train >= best_score_train:
                best_score_train = mean_score_train
                best_i_train = i
                print(f'New best train ({i}).')
                best_score_str_train = score_str
                # save_modelhyperparams(zlinridge, param_gridlin, save_dir, model_type, 'zlinear_train',
                #                       scaler_used=False)
            if mean_score >= best_score:
                best_score = mean_score
                best_i = i
                print(f'New best ({i}).')
                best_score_str = score_str
                save_modelhyperparams(svc, param_gridlin, save_dir, model_type, 'zlinear', scaler_used=False)
        else:
            val_loader = eval_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

            val_inZ = []
            elabels = []
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    inp, labels = data

                    inp = eval_dataset.format_data(inp, device)
                    labels = labels.to(device)
                    log_p, distloss, logdet, out = model(inp, labels)
                    val_inZ.append(out.detach().cpu().numpy())
                    elabels.append(labels.detach().cpu().numpy())
            val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(eval_dataset), -1)
            elabels = np.concatenate(elabels, axis=0)

            svc_score = svc.score(val_inZ, elabels)
            svc_scores.append(svc_score)

            svc_score_train = svc.score(Z, tlabels)

            save_fig(Z, tlabels, size=20, save_path=f'{save_dir}/Z_space_{i}')
            save_fig(val_inZ, elabels, size=20, save_path=f'{save_dir}/Z_space_val_{i}')

            score_str = f'Our approach ({i}) : {svc_score}, (train score : {svc_score_train})'
            print(score_str)

            print(f'Our approach ({i}) : {svc_score}')
            if svc_score >= best_score:
                best_score = svc_score
                best_i = i
                print(f'New best ({i}).')
                best_score_str = score_str
                save_modelhyperparams(svc, param_gridlin, save_dir, model_type, 'zlinear', scaler_used=False)

            if svc_score_train >= best_score_train:
                best_score_train = svc_score_train
                best_i_train = i
                print(f'New best train ({i}).')
                best_score_str_train = score_str
                save_modelhyperparams(svc, param_gridlin, save_dir, model_type, 'zlinear_train', scaler_used=False)

        model.del_model_from_gpu()

    model_loading_params = t_model_params[best_i]
    model_single = load_model_from_params(model_loading_params, full_dataset)

    if not cus_load_func:
        # model = WrappedModel(model_single)
        model = model_single
        model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
        # model = model_single
    else:
        model = cus_load_func(model_single, model_loading_params['loading_path'])
    torch.save(
        model.state_dict(), f"{save_dir}/best_classification_model.pth"
    )

    lines = [best_score_str, '\n', best_score_str_train]
    with open(f"{save_dir}/res.txt", 'w') as f:
        f.writelines(lines)

    if with_train:
        model_loading_params = t_model_params[best_i_train]
        model_single = load_model_from_params(model_loading_params, full_dataset)

        if not cus_load_func:
            # model = WrappedModel(model_single)
            model = model_single
            model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
        else:
            model = cus_load_func(model_single, model_loading_params['loading_path'])
        torch.save(
            model.state_dict(), f"{save_dir}/best_classification_train_model.pth"
        )

    return best_i


def evaluate_regression(t_model_params, train_dataset, eval_dataset, full_dataset, save_dir, device, with_train=False,
                        reg_use_var=False, cus_load_func=None, batch_size=200):
    zridge_scores = []

    # Our approach
    best_score = math.inf
    best_i = 0
    best_score_train = math.inf
    best_i_train = 0
    best_score_str_train = ''
    best_score_str = ''

    # save_fig(train_dataset.X, train_dataset.true_labels, size=10, save_path=f'{save_dir}/X_space')
    # save_fig(val_dataset.X, val_dataset.true_labels, size=10, save_path=f'{save_dir}/X_space_val')

    if isinstance(train_dataset, GraphDataset):  # Permutations for Graphs
        train_dataset.permute_graphs_in_dataset()
        eval_dataset.permute_graphs_in_dataset()

    start_from = None
    # start_from = 289
    for i, model_loading_params in enumerate(t_model_params):
        if start_from is not None and i < start_from:
            continue
        model_single = load_model_from_params(model_loading_params, full_dataset)
        try:
            if not cus_load_func:
                # model_single = WrappedModel(model_single)
                model = model_single
                model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
                # model = model_single.module
            else:
                model = cus_load_func(model_single, model_loading_params['loading_path'])
        except:
            print(f'Exception while loading model {i}.')
            continue
        model = model.to(device)
        model.eval()

        loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

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

        param_gridlin = [{'Ridge__alpha': np.concatenate((np.logspace(-5, 2, 11), np.array([1])))}]
        model_type = ('Ridge', Ridge())
        scaler = False
        zlinridge = learn_or_load_modelhyperparams(Z, tlabels, 'zlinear', param_gridlin, save_dir,
                                                   model_type=model_type, scaler=scaler, save=False,
                                                   force_train=True)
        if isinstance(train_dataset, GraphDataset):  # Permutations for Graphs
            n_permutation = 10
            scores = []
            scores_train = []
            for n_perm in range(n_permutation):
                eval_dataset.permute_graphs_in_dataset()

                val_loader = eval_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

                val_inZ = []
                elabels = []
                with torch.no_grad():
                    for j, data in enumerate(val_loader):
                        inp, labels = data

                        inp = eval_dataset.format_data(inp, device)
                        labels = labels.to(device)
                        log_p, distloss, logdet, out = model(inp, labels)
                        val_inZ.append(out.detach().cpu().numpy())
                        elabels.append(labels.detach().cpu().numpy())
                val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(eval_dataset), -1)
                elabels = np.concatenate(elabels, axis=0)

                y_pred = zlinridge.predict(val_inZ)
                zridge_score = (np.power((y_pred - elabels), 2)).mean()
                zridge_scores.append(zridge_score)

                # Train score
                y_pred_train = zlinridge.predict(Z)
                zridge_score_train = (np.power((y_pred_train - tlabels), 2)).mean()
                scores.append(zridge_score)
                scores_train.append(zridge_score_train)
            mean_score = np.mean(scores)
            mean_score_train = np.mean(scores_train)
            score_str = f'Our approach ({i}) : {scores}, (train score : {scores_train}) \n' \
                        f'Mean Scores: {mean_score}, (train score : {mean_score_train})'
            print(score_str)
            # if zridge_score >= best_score:
            if mean_score_train <= best_score_train:
                best_score_train = mean_score_train
                best_i_train = i
                print(f'New best train ({i}).')
                best_score_str_train = score_str
                # save_modelhyperparams(zlinridge, param_gridlin, save_dir, model_type, 'zlinear_train',
                #                       scaler_used=False)
            if mean_score <= best_score:
                best_score = mean_score
                best_i = i
                print(f'New best ({i}).')
                best_score_str = score_str
                # save_modelhyperparams(zlinridge, param_gridlin, save_dir, model_type, 'zlinear', scaler_used=False)
        else:
            val_loader = eval_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

            val_inZ = []
            elabels = []
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    inp, labels = data

                    inp = eval_dataset.format_data(inp, device)
                    labels = labels.to(device)
                    log_p, distloss, logdet, out = model(inp, labels)
                    val_inZ.append(out.detach().cpu().numpy())
                    elabels.append(labels.detach().cpu().numpy())
            val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(eval_dataset), -1)
            elabels = np.concatenate(elabels, axis=0)

            # zridge_score = zlinridge.score(val_inZ, elabels)
            y_pred = zlinridge.predict(val_inZ)
            zridge_score = (np.power((y_pred - elabels), 2)).mean()
            zridge_scores.append(zridge_score)

            # Analytic predictions:
            # device_p = device
            # device_p = 'cpu'
            # Zt = torch.from_numpy(Z).to(device_p)
            # val_inZt = torch.from_numpy(val_inZ).to(device_p)
            # tlabelst = torch.from_numpy(tlabels).float().to(device_p)
            # norm_Z = Zt / torch.sum(Zt, dim=1, keepdim=True)
            # norm_val_inZ = val_inZt / torch.sum(val_inZt, dim=1, keepdim=True)
            # alphas = torch.matmul(
            #     torch.inverse(torch.matmul(norm_Z, norm_Z.transpose(1, 0)) + 0.01 * torch.eye(Zt.shape[0]).to(device_p)),
            #     tlabelst)
            # pred = torch.matmul(torch.sum(alphas.unsqueeze(1) * norm_Z, dim=0),
            #                     norm_val_inZ.transpose(1, 0)).detach().cpu().numpy()
            # test analytic phi
            # A = torch.pow(norm_Z, 2) + 0.1
            # left_inv = torch.matmul(torch.inverse(torch.matmul(A.transpose(1, 0), A)), A.transpose(1, 0))
            # weights = left_inv @ (2 * tlabelst.unsqueeze(1) * (norm_Z @ torch.ones((norm_Z.shape[1], 1))))
            # pred = torch.matmul(norm_val_inZ, weights).detach().cpu().numpy()
            # score = (np.power((pred - elabels), 2)).mean()

            # TEST #
            # if i == 47:
            #     numpy.save(f'{save_dir}/test_save_Z', Z)
            #     numpy.save(f'{save_dir}/test_save_valinZ', val_inZ)
            #     numpy.save(f'{save_dir}/test_save_y', elabels)
            #     numpy.save(f'{save_dir}/test_save_predy', y_pred)

            # Train score
            # zridge_score_train = zlinridge.score(Z, tlabels)
            y_pred_train = zlinridge.predict(Z)
            zridge_score_train = (np.power((y_pred_train - tlabels), 2)).mean()

            save_fig(Z, tlabels, size=20, save_path=f'{save_dir}/Z_space_{i}')
            save_fig(val_inZ, elabels, size=20, save_path=f'{save_dir}/Z_space_val_{i}')

            # TEST project between the means
            from utils.testing import project_between
            means = model.means.detach().cpu().numpy()
            proj, dot_val = project_between(val_inZ, means[0], means[1])
            # pred = ((proj - means[1]) / (means[0] - means[1])) * (
            #         model.label_max - model.label_min) + model.label_min
            # pred = pred.mean(axis=1)
            pred = dot_val.squeeze() * (model.label_max - model.label_min) + model.label_min
            projection_score = np.power((pred - elabels), 2).mean()
            # score_str = f'Our approach ({i}) : {zridge_score}, (train score : {zridge_score_train}), (projection) : {projection_score}, (analytic pred): {score}'
            score_str = f'Our approach ({i}) : {zridge_score}, (train score : {zridge_score_train}), (projection) : {projection_score}'
            print(score_str)
            # if zridge_score >= best_score:
            if zridge_score_train <= best_score_train:
                best_score_train = zridge_score_train
                best_i_train = i
                print(f'New best train ({i}).')
                best_score_str_train = score_str
                save_modelhyperparams(zlinridge, param_gridlin, save_dir, model_type, 'zlinear_train',
                                      scaler_used=False)
            if zridge_score <= best_score:
                best_score = zridge_score
                best_i = i
                print(f'New best ({i}).')
                best_score_str = score_str
                save_modelhyperparams(zlinridge, param_gridlin, save_dir, model_type, 'zlinear', scaler_used=False)

        model.del_model_from_gpu()

    model_loading_params = t_model_params[best_i]
    model_single = load_model_from_params(model_loading_params, full_dataset)

    if not cus_load_func:
        # model = WrappedModel(model_single)
        model = model_single
        model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
    else:
        model = cus_load_func(model_single, model_loading_params['loading_path'])
    torch.save(
        model.state_dict(), f"{save_dir}/best_regression_model.pth"
    )

    lines = [best_score_str, '\n', best_score_str_train]
    with open(f"{save_dir}/res.txt", 'w') as f:
        f.writelines(lines)

    if with_train:
        model_loading_params = t_model_params[best_i_train]
        model_single = load_model_from_params(model_loading_params, full_dataset)

        if not cus_load_func:
            # model = WrappedModel(model_single)
            model = model_single
            model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
        else:
            model = cus_load_func(model_single, model_loading_params['loading_path'])
        torch.save(
            model.state_dict(), f"{save_dir}/best_regression_train_model.pth"
        )

    return best_i


def evaluate_distances(t_model_params, train_dataset, val_dataset, full_dataset, gaussian_params, how_much, device,
                       save_dir, noise_type='gaussian', eval_gaussian_std=.1, cus_load_func=None, batch_size=200,
                       proj_type='zpca_l'):
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

    best_score = math.inf
    best_i = 0
    start_from = None
    # start_from = 289
    for i, model_loading_params in enumerate(t_model_params):
        if start_from is not None and i < start_from:
            continue
        model_single = load_model_from_params(model_loading_params, full_dataset)

        try:
            if not cus_load_func:
                # model_single = WrappedModel(model_single)
                model = model_single
                model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
                # model = model_single.module
            else:
                model = cus_load_func(model_single, model_loading_params['loading_path'])
        except:
            print(f'Exception while loading model {i}.')
            continue
        model = model.to(device)
        model.eval()

        z_shape = model.calc_last_z_shape(train_dataset.in_size)

        loader = train_dataset_noised.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

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

        projs = []
        ordered_val = []
        ordered_elabels = []
        for j, gaussian_param in enumerate(gaussian_params):
            gmean = model.means[j].detach().cpu().numpy()
            gp = gaussian_param[1:-1]
            label = gaussian_param[-1]

            indexes = np.where(elabels == label)[0]
            val_Z_lab = val_inZ[indexes]
            val_lab = val_dataset.X[indexes]
            # val_noised_lab = val_noised[indexes]

            ordered_elabels.append(np.array([label for _ in range(indexes.shape[0])]))

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
            ordered_val.append(val_lab)
        ordered_val = np.concatenate(ordered_val, axis=0)
        proj = np.concatenate(projs, axis=0)

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

        print(f'Our approach ({i}) : {mean_dist}')
        if mean_dist <= best_score:
            best_score = mean_dist
            best_i = i
        if isinstance(val_dataset, SimpleDataset):
            save_fig(ordered_val, all_res, np.concatenate(ordered_elabels, axis=0), label_max=1, size=5,
                     save_path=f'{save_dir}/projection_zpca_model{i}')

    print(best_i)
    model_loading_params = t_model_params[best_i]
    model_single = load_model_from_params(model_loading_params, full_dataset)

    if not cus_load_func:
        # model = WrappedModel(model_single)
        model = model_single
        model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
    else:
        model = cus_load_func(model_single, model_loading_params['loading_path'])
    torch.save(
        model.state_dict(), f"{save_dir}/best_projection_model.pth"
    )


def visualize_dataset(dataset, train_dataset, val_dataset, model, save_dir, device):
    save_path = f'{save_dir}/visualize'
    create_folder(save_path)
    hist_train = np.histogram(train_dataset.true_labels)
    hist_val = np.histogram(val_dataset.true_labels, bins=hist_train[1])
    idxs_train = np.argsort(train_dataset.true_labels)
    idxs_val = np.argsort(val_dataset.true_labels)
    done_train = 0
    done_val = 0
    model.eval()

    import scipy
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(dataset.X)
    X_pca = pca.transform(dataset.X)

    save_fig(X_pca, dataset.true_labels, f'{save_path}/PCA_X.png', size=20)

    minx = model.means[:, 0].detach().cpu().numpy().min()
    maxx = model.means[:, 0].detach().cpu().numpy().max()
    diff = maxx - minx
    minx -= diff / 2
    maxx += diff / 2
    miny = model.means[:, 1].detach().cpu().numpy().min()
    maxy = model.means[:, 1].detach().cpu().numpy().max()
    diff = maxy - miny
    miny -= diff / 2
    maxy += diff / 2

    for i, nb_class in enumerate(hist_train[0]):
        nb_class_val = hist_val[0][i]
        X_train = train_dataset.X[idxs_train[done_train:done_train + nb_class]]
        X_val = val_dataset.X[idxs_val[done_val:done_val + nb_class_val]]
        labels_train = train_dataset.true_labels[idxs_train[done_train:done_train + nb_class]]
        labels_val = val_dataset.true_labels[idxs_val[done_val:done_val + nb_class_val]]
        X = np.concatenate((X_train, X_val), axis=0)
        dist = scipy.spatial.distance.cdist(X, X)
        print(f'Distances({i}) : \n' + str(dist))
        print(f'Mean Distances({i}) : \n' + str(dist.mean(axis=0)))
        print(f'Nval:{nb_class_val}')
        labels = np.concatenate((labels_train, np.ones_like(labels_val) * -100), axis=0)
        save_fig(X, labels, f'{save_path}/X_{i}.png', size=20,
                 limits=(dataset.X[:, 0].min(), dataset.X[:, 0].max(), dataset.X[:, 1].min(), dataset.X[:, 0].max()),
                 rangelabels=(-100, model.label_max))

        pca = PCA(n_components=2)
        pca.fit(X)
        X_pca = pca.transform(X)

        save_fig(X_pca, labels, f'{save_path}/PCA_X_{i}.png', size=20, rangelabels=(-100, model.label_max))

        with torch.no_grad():
            inp = torch.from_numpy(X.reshape(X.shape[0], -1)).float().to(device)
            labels = torch.from_numpy(labels).to(device)
            log_p, distloss, logdet, out = model(inp, labels)
            Z = out.detach().cpu().numpy()
        Z = Z.reshape(Z.shape[0], -1)
        save_fig(Z, labels.detach().cpu().numpy(), f'{save_path}/Z_{i}.png', size=20, limits=(minx, maxx, miny, maxy),
                 rangelabels=(-100, model.label_max))
        done_train += nb_class
        done_val += nb_class_val


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]

    # Retrieve parameters from name
    config = retrieve_params_from_name(folder_name, model_type)

    # DATASET #
    dataset = load_dataset(args, dataset_name, model_type, to_evaluate=True, add_feature=config['add_feature'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config['label'] is not None:
        dataset.reduce_dataset('one_class', label=config['label'])

    os.chdir(args.folder)
    saves = []
    # for file in glob.glob("model_*.pt"):
    #     saves.append(file)
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
    if args.reduce_test_dataset_size is not None:
        train_dataset = train_dataset.duplicate()
        print('Train dataset reduced in order to accelerate. (stratified)')
        train_dataset.reduce_dataset_ratio(args.reduce_test_dataset_size, stratified=True)

    # dim_per_label, n_dim = dataset.get_dim_per_label(return_total_dim=True)
    n_dim = dataset.get_n_dim()
    dim_per_label = config['dim_per_label']

    t_model_params = []
    for save_path in saves:
        # initialize gaussian params
        if config['uniform_eigval']:
            var_type = 'uniform'
        elif config['gaussian_eigval'] is not None:
            var_type = 'gaussian'
        elif config['fixed_eigval'] is not None:
            var_type = 'manual'
        else:
            assert 'Unknown var_type !'
        gaussian_params = initialize_gaussian_params(args, dataset, var_type, config['mean_of_eigval'],
                                                     dim_per_label, 'isotrope' in folder_name,
                                                     config['split_graph_dim'], fixed_eigval=config['fixed_eigval'],
                                                     gaussian_eigval=config['gaussian_eigval'],
                                                     add_feature=config['add_feature'])

        learn_mean = 'lmean' in folder_name
        model_loading_params = {'model': model_type, 'n_dim': n_dim, 'gaussian_params': gaussian_params,
                                'device': device, 'loading_path': save_path, 'learn_mean': learn_mean}
        t_model_params.append(prepare_model_loading_params(model_loading_params, config, dataset, model_type))

    save_dir = './save'
    create_folder(save_dir)

    if args.eval_type == 'classification':
        evaluate_classification(t_model_params, train_dataset, val_dataset, full_dataset=dataset, save_dir=save_dir,
                                device=device, with_train=True, batch_size=args.batch_size)
    elif args.eval_type == 'regression':
        evaluate_regression(t_model_params, train_dataset, val_dataset, full_dataset=dataset,
                            save_dir=save_dir, device=device, with_train=True,
                            reg_use_var=config['reg_use_var'])
    elif args.eval_type == 'projection':

        noise_type = 'gaussian'
        noise_std = .1
        proj_type = 'gp'
        batch_size = 16

        print(f'(noise_type, noise_std, proj_type, batch_size) are set manually to '
              f'({noise_type},{noise_std},{proj_type},{batch_size}).')
        n_principal_dim = np.count_nonzero(gaussian_params[0][-2] > 1)
        evaluate_distances(t_model_params, train_dataset, val_dataset, dataset, gaussian_params, n_principal_dim,
                           device,
                           noise_type=noise_type, eval_gaussian_std=noise_std, save_dir=save_dir, proj_type=proj_type,
                           batch_size=batch_size)


if __name__ == '__main__':
    parser = testing_arguments()
    choices = ['classification', 'projection', 'regression']
    best_model_choices = ['classification', 'projection', 'regression']
    for choice in best_model_choices.copy():
        best_model_choices.append(choice + '_train')
    parser = testing_arguments()
    parser.add_argument('--eval_type', type=str, default='classification', choices=choices, help='evaluation type')
    parser.add_argument('--model_to_use', type=str, default='classification', choices=best_model_choices,
                        help='what best model to use for the evaluation')
    args = parser.parse_args()
    args.seed = 0

    main(args)
