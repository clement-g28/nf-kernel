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

from utils.testing import testing_arguments, save_modelhyperparams, retrieve_params_from_name, \
    learn_or_load_modelhyperparams, initialize_gaussian_params, prepare_model_loading_params, load_split_dataset, \
    load_model_from_params


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
    # start_from = 100
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
        model_single = load_model_from_params(model_loading_params, full_dataset)

        if not cus_load_func:
            model = WrappedModel(model_single)
            model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
        else:
            model = cus_load_func(model_single, model_loading_params['loading_path'])
        torch.save(
            model.state_dict(), f"{save_dir}/best_classification_train_model.pth"
        )

    return best_i


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]
    # DATASET #
    dataset = load_dataset(args, dataset_name, model_type, to_evaluate=True)

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
    print('Train dataset reduced in order to accelerate. (stratified)')
    train_dataset.reduce_dataset_ratio(0.5, stratified=True)
    # val_dataset.reduce_dataset_ratio(0.1, stratified=True)

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

    evaluate_classification(t_model_params, train_dataset, val_dataset, full_dataset=dataset, save_dir=save_dir,
                            device=device, with_train=True, batch_size=10)


if __name__ == '__main__':
    parser = testing_arguments()
    args = parser.parse_args()
    args.seed = 0

    main(args)
