import numpy
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
from utils.testing import save_modelhyperparams
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments, seqflow_arguments, graphnvp_arguments

from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model
from utils.models import IMAGE_MODELS, SIMPLE_MODELS, GRAPH_MODELS


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

    # TEST
    start_from = None
    # start_from = 289
    for i, model_loading_params in enumerate(t_model_params):
        if start_from is not None and i < start_from:
            continue
        if model_loading_params['model'] == 'cglow':
            # Load model
            model_single = CGlow(model_loading_params['n_channel'], model_loading_params['n_flow'],
                                 model_loading_params['n_block'], affine=model_loading_params['affine'],
                                 conv_lu=model_loading_params['conv_lu'],
                                 gaussian_params=model_loading_params['gaussian_params'],
                                 device=model_loading_params['device'], learn_mean=model_loading_params['learn_mean'])
        elif model_loading_params['model'] == 'seqflow':
            model_single = load_seqflow_model(model_loading_params['n_dim'], model_loading_params['n_flow'],
                                              gaussian_params=model_loading_params['gaussian_params'],
                                              learn_mean=model_loading_params['learn_mean'], dataset=full_dataset)

        elif model_loading_params['model'] == 'ffjord':
            model_single = load_ffjord_model(model_loading_params['ffjord_args'], model_loading_params['n_dim'],
                                             gaussian_params=model_loading_params['gaussian_params'],
                                             learn_mean=model_loading_params['learn_mean'], dataset=full_dataset)
        elif model_loading_params['model'] == 'moflow':
            args_moflow, _ = moflow_arguments().parse_known_args()
            model_single = load_moflow_model(model_loading_params['moflow_args'],
                                             gaussian_params=model_loading_params['gaussian_params'],
                                             learn_mean=model_loading_params['learn_mean'], reg_use_var=reg_use_var,
                                             dataset=full_dataset)
        else:
            assert False, 'unknown model type!'
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
                inp = train_dataset.format_data(inp, None, None, device)
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

                        inp = eval_dataset.format_data(inp, None, None, device)
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
            loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

            Z = []
            tlabels = []
            with torch.no_grad():
                for j, data in enumerate(loader):
                    inp, labels = data
                    inp = train_dataset.format_data(inp, None, None, device)
                    labels = labels.to(device)
                    log_p, distloss, logdet, out = model(inp, labels)
                    Z.append(out.detach().cpu().numpy())
                    tlabels.append(labels.detach().cpu().numpy())
            Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)
            tlabels = np.concatenate(tlabels, axis=0)

            # Learn Ridge
            # zlinridge = make_pipeline(StandardScaler(), KernelRidge(kernel='linear', alpha=0.1))
            # zlinridge = Ridge(alpha=1.0)
            # zlinridge.fit(Z, tlabels)
            param_gridlin = [{'Ridge__alpha': np.concatenate((np.logspace(-5, 2, 11), np.array([1])))}]
            model_type = ('Ridge', Ridge())
            scaler = False
            zlinridge = learn_or_load_modelhyperparams(Z, tlabels, 'zlinear', param_gridlin, save_dir,
                                                       model_type=model_type, scaler=scaler, save=False,
                                                       force_train=True)
            # kernel_name = 'linear'
            # param_gridlin = [{'Ridge__kernel': [kernel_name], 'Ridge__alpha': np.linspace(0, 10, 11)}]
            # zlinridge = learn_or_load_modelhyperparams(Z, tlabels, kernel_name, param_gridlin, save_dir,
            #                                            model_type=('Ridge', KernelRidge()), scaler=False)

            val_loader = eval_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

            val_inZ = []
            elabels = []
            with torch.no_grad():
                for j, data in enumerate(val_loader):
                    inp, labels = data

                    inp = eval_dataset.format_data(inp, None, None, device)
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

    model_loading_params = t_model_params[best_i]
    if model_loading_params['model'] == 'cglow':
        # Load model
        model_single = CGlow(model_loading_params['n_channel'], model_loading_params['n_flow'],
                             model_loading_params['n_block'], affine=model_loading_params['affine'],
                             conv_lu=model_loading_params['conv_lu'],
                             gaussian_params=model_loading_params['gaussian_params'],
                             device=model_loading_params['device'], learn_mean=model_loading_params['learn_mean'])
    elif model_loading_params['model'] == 'seqflow':
        model_single = load_seqflow_model(model_loading_params['n_dim'], model_loading_params['n_flow'],
                                          gaussian_params=model_loading_params['gaussian_params'],
                                          learn_mean=model_loading_params['learn_mean'], dataset=full_dataset)

    elif model_loading_params['model'] == 'ffjord':
        model_single = load_ffjord_model(model_loading_params['ffjord_args'], model_loading_params['n_dim'],
                                         gaussian_params=model_loading_params['gaussian_params'],
                                         learn_mean=model_loading_params['learn_mean'], dataset=full_dataset)
    elif model_loading_params['model'] == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        model_single = load_moflow_model(model_loading_params['moflow_args'],
                                         gaussian_params=model_loading_params['gaussian_params'],
                                         learn_mean=model_loading_params['learn_mean'], reg_use_var=reg_use_var,
                                         dataset=full_dataset)
    else:
        umodel_type = model_loading_params['model']
        assert False, f'unknown model type! ({umodel_type})'

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
        if model_loading_params['model'] == 'cglow':
            # Load model
            model_single = CGlow(model_loading_params['n_channel'], model_loading_params['n_flow'],
                                 model_loading_params['n_block'], affine=model_loading_params['affine'],
                                 conv_lu=model_loading_params['conv_lu'],
                                 gaussian_params=model_loading_params['gaussian_params'],
                                 device=model_loading_params['device'], learn_mean=model_loading_params['learn_mean'])
        elif model_loading_params['model'] == 'seqflow':
            model_single = load_seqflow_model(model_loading_params['n_dim'], model_loading_params['n_flow'],
                                              gaussian_params=model_loading_params['gaussian_params'],
                                              learn_mean=model_loading_params['learn_mean'], dataset=full_dataset)

        elif model_loading_params['model'] == 'ffjord':
            model_single = load_ffjord_model(model_loading_params['ffjord_args'], model_loading_params['n_dim'],
                                             gaussian_params=model_loading_params['gaussian_params'],
                                             learn_mean=model_loading_params['learn_mean'], dataset=full_dataset)
        elif model_loading_params['model'] == 'moflow':
            args_moflow, _ = moflow_arguments().parse_known_args()
            model_single = load_moflow_model(model_loading_params['moflow_args'],
                                             gaussian_params=model_loading_params['gaussian_params'],
                                             learn_mean=model_loading_params['learn_mean'], reg_use_var=reg_use_var,
                                             dataset=full_dataset)
        else:
            model_type = model_loading_params['model']
            assert False, f'unknown model type! ({model_type})'

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

    os.chdir(args.folder)
    saves = []
    for file in glob.glob("model_*.pt"):
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
            gaussian_params = initialize_gaussian_params(dataset, eigval_list, isotrope='isotrope' in folder_name,
                                                         dim_per_label=dim_per_label, fixed_eigval=fixed_eigval)
        else:
            if args.method == 0:
                gaussian_params = initialize_regression_gaussian_params(dataset, eigval_list,
                                                                        isotrope='isotrope' in folder_name,
                                                                        dim_per_label=dim_per_label,
                                                                        fixed_eigval=fixed_eigval)
            elif args.method == 1:
                gaussian_params = initialize_tmp_regression_gaussian_params(dataset, eigval_list)
            elif args.method == 2:
                gaussian_params = initialize_tmp_regression_gaussian_params(dataset, eigval_list, ones=True)
            else:
                assert False, 'no method selected'

        learn_mean = 'lmean' in folder_name
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

    best_i = evaluate_regression(t_model_params, train_dataset, val_dataset, full_dataset=dataset, save_dir=save_dir,
                                 device=device, with_train=True, reg_use_var=reg_use_var)

    model_loading_params = t_model_params[best_i]
    if model_loading_params['model'] == 'cglow':
        # Load model
        model_single = CGlow(model_loading_params['n_channel'], model_loading_params['n_flow'],
                             model_loading_params['n_block'], affine=model_loading_params['affine'],
                             conv_lu=model_loading_params['conv_lu'],
                             gaussian_params=model_loading_params['gaussian_params'],
                             device=model_loading_params['device'], learn_mean=model_loading_params['learn_mean'])
    elif model_loading_params['model'] == 'seqflow':
        model_single = load_seqflow_model(model_loading_params['n_dim'], model_loading_params['n_flow'],
                                          gaussian_params=model_loading_params['gaussian_params'],
                                          learn_mean=model_loading_params['learn_mean'], reg_use_var=reg_use_var,
                                          dataset=dataset)

    elif model_loading_params['model'] == 'ffjord':
        model_single = load_ffjord_model(model_loading_params['ffjord_args'], model_loading_params['n_dim'],
                                         gaussian_params=model_loading_params['gaussian_params'],
                                         learn_mean=model_loading_params['learn_mean'], reg_use_var=reg_use_var,
                                         dataset=dataset)
    elif model_loading_params['model'] == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        model_single = load_moflow_model(model_loading_params['moflow_args'],
                                         gaussian_params=model_loading_params['gaussian_params'],
                                         learn_mean=model_loading_params['learn_mean'], reg_use_var=reg_use_var,
                                         dataset=dataset)

    else:
        assert False, 'unknown model type!'
    model_single = WrappedModel(model_single)
    model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
    model = model_single.module
    model = model.to(device)
    model.eval()
    # visualize_dataset(dataset, train_dataset, val_dataset, model, save_dir)
