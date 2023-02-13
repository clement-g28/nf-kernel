import numpy as np
import math
import torch
import glob
import os
import re

from utils.custom_glow import CGlow, WrappedModel
from utils.models import IMAGE_MODELS, SIMPLE_MODELS, GRAPH_MODELS
from utils.dataset import ImDataset, SimpleDataset, GraphDataset, RegressionGraphDataset, ClassificationGraphDataset, \
    SIMPLE_DATASETS, SIMPLE_REGRESSION_DATASETS, IMAGE_DATASETS, GRAPH_REGRESSION_DATASETS, \
    GRAPH_CLASSIFICATION_DATASETS

from utils.utils import set_seed, create_folder, initialize_gaussian_params, initialize_regression_gaussian_params, \
    save_fig, initialize_tmp_regression_gaussian_params

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.testing import testing_arguments
from utils.testing import save_modelhyperparams
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments, seqflow_arguments, graphnvp_arguments

from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model
from utils.testing import learn_or_load_modelhyperparams


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
                                             learn_mean=model_loading_params['learn_mean'],
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

        # Learn SVC
        param_gridlin = [{'SVC__kernel': ['linear'], 'SVC__C': np.array([1])}]
        # param_gridlin = [{'SVC__kernel': ['linear'], 'SVC__C': np.concatenate((np.logspace(-5, 2, 11), np.array([1])))}]
        model_type = ('SVC', SVC())
        scaler = False
        svc = learn_or_load_modelhyperparams(Z, tlabels, 'zlinear', param_gridlin, save_dir,
                                             model_type=model_type, scaler=scaler, save=False,
                                             force_train=True)
        # svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        # svc.fit(Z, tlabels)

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
                # save_modelhyperparams(zlinridge, param_gridlin, save_dir, model_type, 'zlinear', scaler_used=False)
        else:
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
                                         learn_mean=model_loading_params['learn_mean'],
                                         dataset=full_dataset)
    else:
        assert False, 'unknown model type!'

    if not cus_load_func:
        model = WrappedModel(model_single)
        model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
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
                                             learn_mean=model_loading_params['learn_mean'],
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
            model.state_dict(), f"{save_dir}/best_classifiation_train_model.pth"
        )

    return best_i


if __name__ == '__main__':
    parser = testing_arguments()
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
            mean_of_eigval = in_split[0].replace("eigvalgaussian", "")
            mean_of_eigval = float(mean_of_eigval.replace("-", "."))
            std_value = float(str(in_split[-1]).replace('-', '.'))
            gaussian_eigval = [0.0, std_value]
            print(f'Flow trained with gaussian eigenvalues params: {mean_of_eigval},{gaussian_eigval}')
        elif 'dimperlab' in split:
            dpl_split = split
            dim_per_label = int(dpl_split.replace("dimperlab", ""))
            print(f'Flow trained with dimperlab: {dim_per_label}')

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

    evaluate_classification(t_model_params, train_dataset, val_dataset, full_dataset=dataset, save_dir=save_dir,
                            device=device, with_train=True, batch_size=10)
