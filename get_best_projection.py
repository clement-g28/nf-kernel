import numpy as np
import math
import torch
import glob
import os

from utils.custom_glow import WrappedModel

from utils.utils import set_seed, create_folder, load_dataset

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from utils.testing import testing_arguments, noise_data, save_fig, retrieve_params_from_name, \
    learn_or_load_modelhyperparams, initialize_gaussian_params, prepare_model_loading_params, load_split_dataset, \
    load_model_from_params


def evaluate_distances(t_model_params, train_dataset, val_dataset, full_dataset, gaussian_params, how_much, device,
                       noise_type='gaussian', eval_gaussian_std=.1, cus_load_func=None, batch_size=200):
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
        for gaussian_param in gaussian_params:
            label = gaussian_param[-1]

            indexes = np.where(elabels == label)[0]
            val_Z_lab = val_inZ[indexes]
            val_lab = val_dataset.X[indexes]
            # val_noised_lab = val_noised[indexes]

            ordered_elabels.append(np.array([label for _ in range(indexes.shape[0])]))

            # Z-PCA
            Z_lab = Z[np.where(tlabels == label)]
            pca = PCA(n_components=how_much)
            pca.fit(Z_lab)
            pca_projection = pca.transform(val_Z_lab)
            proj = pca.inverse_transform(pca_projection)

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

        save_fig(ordered_val, all_res, np.concatenate(ordered_elabels, axis=0), label_max=1, size=5,
                 save_path=f'{save_dir}/projection_zpca_model{i}')

    print(best_i)
    model_loading_params = t_model_params[best_i]
    model_single = load_model_from_params(model_loading_params, full_dataset)

    if not cus_load_func:
        model = WrappedModel(model_single)
        model.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
    else:
        model = cus_load_func(model_single, model_loading_params['loading_path'])
    torch.save(
        model.state_dict(), f"{save_dir}/best_projection_model.pth"
    )


if __name__ == '__main__':
    parser = testing_arguments()
    args = parser.parse_args()

    set_seed(0)

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

    n_principal_dim = np.count_nonzero(gaussian_params[0][-2] > 1)
    evaluate_distances(t_model_params, train_dataset, val_dataset, dataset, gaussian_params, n_principal_dim, device,
                       noise_type='gaussian', eval_gaussian_std=.1)
