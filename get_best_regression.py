import numpy as np
import math
import torch
import glob
import os
import re

from utils.custom_glow import CGlow, WrappedModel
from utils.dataset import ImDataset, SimpleDataset

from utils.utils import set_seed, create_folder, initialize_gaussian_params, initialize_regression_gaussian_params, \
    save_fig, initialize_tmp_regression_gaussian_params

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from utils.testing import testing_arguments
from utils.testing import learn_or_load_modelhyperparams
from utils.training import ffjord_arguments

from utils.toy_models import load_seqflow_model, load_ffjord_model


def evaluate_regression(t_model_params, train_dataset, eval_dataset, full_dataset, save_dir, device):
    zridge_scores = []

    # Our approach
    best_score = 0
    best_score = math.inf
    best_i = 0

    save_fig(train_dataset.X, train_dataset.true_labels, size=10, save_path=f'{save_dir}/X_space')
    save_fig(val_dataset.X, val_dataset.true_labels, size=10, save_path=f'{save_dir}/X_space_val')
    # Get z_val for zpca
    for i, model_loading_params in enumerate(t_model_params):
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

        else:
            assert False, 'unknown model type!'
        model_single = WrappedModel(model_single)
        model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
        model = model_single.module
        model = model.to(device)
        model.eval()

        batch_size = 20
        loader = train_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

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

        # Learn Ridge
        # zlinridge = make_pipeline(StandardScaler(), KernelRidge(kernel='linear', alpha=0.1))
        zlinridge = Ridge(alpha=1.0)
        zlinridge.fit(Z, tlabels)
        # kernel_name = 'linear'
        # param_gridlin = [{'Ridge__kernel': [kernel_name], 'Ridge__alpha': np.linspace(0, 10, 11)}]
        # zlinridge = learn_or_load_modelhyperparams(Z, tlabels, kernel_name, param_gridlin, save_dir,
        #                                            model_type=('Ridge', KernelRidge()), scaler=False)

        val_loader = eval_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

        val_inZ = []
        elabels = []
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                inp = data[0].float().to(device)
                labels = data[1].to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                val_inZ.append(out.detach().cpu().numpy())
                elabels.append(labels.detach().cpu().numpy())
        val_inZ = np.concatenate(val_inZ, axis=0).reshape(len(eval_dataset), -1)
        elabels = np.concatenate(elabels, axis=0)

        # zridge_score = zlinridge.score(val_inZ, elabels)
        y_pred = zlinridge.predict(val_inZ)
        zridge_score = (np.power((y_pred - elabels), 2)).mean()
        zridge_scores.append(zridge_score)

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

        print(
            f'Our approach ({i}) : {zridge_score}, (train score : {zridge_score_train}), (projection) : {projection_score}')
        # if zridge_score >= best_score:
        if zridge_score <= best_score:
            best_score = zridge_score
            best_i = i

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

    else:
        assert False, 'unknown model type!'

    model_single = WrappedModel(model_single)
    model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
    torch.save(
        model_single.state_dict(), f"{save_dir}/best_regression_model.pth"
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
    if dataset_name == 'mnist':
        dataset = ImDataset(dataset_name=dataset_name)
        n_channel = dataset.n_channel
    else:
        dataset = SimpleDataset(dataset_name=dataset_name)
        n_channel = 1

    img_size = dataset.im_size

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

    # Model params
    affine = args.affine
    no_lu = args.no_lu

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

    n_dim = dataset.X[0].shape[0]
    for sh in dataset.X[0].shape[1:]:
        n_dim *= sh

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
        model_loading_params = {'model': model_type, 'n_dim': n_dim, 'n_channel': n_channel, 'n_flow': n_flow,
                                'n_block': n_block, 'affine': affine,
                                'conv_lu': not no_lu, 'gaussian_params': gaussian_params, 'device': device,
                                'loading_path': save_path, 'learn_mean': learn_mean}
        if model_type == 'ffjord':
            args_ffjord, _ = ffjord_arguments().parse_known_args()
            args_ffjord.n_block = n_block
            model_loading_params['ffjord_args'] = args_ffjord

        t_model_params.append(model_loading_params)

    save_dir = './save'
    create_folder(save_dir)

    best_i = evaluate_regression(t_model_params, train_dataset, val_dataset, full_dataset=dataset, save_dir=save_dir,
                                 device=device)

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
                                          learn_mean=model_loading_params['learn_mean'], dataset=dataset)

    elif model_loading_params['model'] == 'ffjord':
        model_single = load_ffjord_model(model_loading_params['ffjord_args'], model_loading_params['n_dim'],
                                         gaussian_params=model_loading_params['gaussian_params'],
                                         learn_mean=model_loading_params['learn_mean'], dataset=dataset)

    else:
        assert False, 'unknown model type!'
    model_single = WrappedModel(model_single)
    model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
    model = model_single.module
    model = model.to(device)
    model.eval()
    visualize_dataset(dataset, train_dataset, val_dataset, model, save_dir)
