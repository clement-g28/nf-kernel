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
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from utils.testing import testing_arguments, save_modelhyperparams, retrieve_params_from_name, \
    learn_or_load_modelhyperparams, initialize_gaussian_params, prepare_model_loading_params, load_split_dataset, \
    load_model_from_params


def evaluate_regression(t_model_params, train_dataset, eval_dataset, full_dataset, save_dir, device, with_train=False,
                        reg_use_var=False, cus_load_func=None):
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
                model_single = WrappedModel(model_single)
                model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
                model = model_single.module
            else:
                model = cus_load_func(model_single, model_loading_params['loading_path'])
        except:
            print(f'Exception while loading model {i}.')
            continue
        model = model.to(device)
        model.eval()

        batch_size = 200
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
        model = WrappedModel(model_single)
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
            model = WrappedModel(model_single)
            model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
        else:
            model = cus_load_func(model_single, model_loading_params['loading_path'])
        torch.save(
            model.state_dict(), f"{save_dir}/best_regression_train_model.pth"
        )

    return best_i


def visualize_dataset(dataset, train_dataset, val_dataset, model, save_dir):
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


if __name__ == '__main__':
    parser = testing_arguments()
    parser.add_argument("--method", default=0, type=int, help='select between [0,1,2]')
    args = parser.parse_args()

    set_seed(0)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]
    # DATASET #
    dataset = load_dataset(args, dataset_name, model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Retrieve parameters from name
    n_block, n_flow, mean_of_eigval, dim_per_label, label, fixed_eigval, uniform_eigval, gaussian_eigval, \
    reg_use_var, split_graph_dim = retrieve_params_from_name(folder_name)

    if label is not None:
        dataset.reduce_dataset('one_class', label=label)

    os.chdir(args.folder)
    saves = []
    for file in glob.glob("model_*.pt"):
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
    # train_dataset.reduce_regression_dataset(0.2, stratified=True)
    # val_dataset.reduce_regression_dataset(0.5, stratified=True)

    n_dim = dataset.get_n_dim()

    if not dim_per_label:
        if not dataset.is_regression_dataset():
            uni = np.unique(dataset.true_labels)
            dim_per_label = math.floor(n_dim / len(uni))
        else:
            dim_per_label = n_dim

    t_model_params = []
    for save_path in saves:
        gaussian_params = initialize_gaussian_params(args, dataset, fixed_eigval, uniform_eigval, gaussian_eigval,
                                                     mean_of_eigval, dim_per_label, 'isotrope' in folder_name,
                                                     split_graph_dim=split_graph_dim)

        learn_mean = 'lmean' in folder_name
        model_loading_params = {'model': model_type, 'n_dim': n_dim, 'n_flow': n_flow,
                                'n_block': n_block, 'gaussian_params': gaussian_params, 'device': device,
                                'loading_path': save_path, 'learn_mean': learn_mean}
        t_model_params.append(prepare_model_loading_params(model_loading_params, dataset, model_type))

    save_dir = './save'
    create_folder(save_dir)

    best_i = evaluate_regression(t_model_params, train_dataset, val_dataset, full_dataset=dataset, save_dir=save_dir,
                                 device=device, with_train=True, reg_use_var=reg_use_var)
