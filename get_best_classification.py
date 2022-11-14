import numpy as np
import math
import torch
import glob
import os
import re

from utils.custom_glow import CGlow, WrappedModel
from utils.dataset import ImDataset, SimpleDataset

from utils.utils import initialize_gaussian_params
from utils.utils import set_seed, create_folder

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.testing import testing_arguments
from utils.training import ffjord_arguments

from utils.toy_models import load_seqflow_model, load_ffjord_model


def evaluate_classification(t_model_params, save_dir, device):
    svc_scores = []

    # Our approach
    eval_dataset = val_dataset

    best_score = 0
    best_i = 0
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
                                              learn_mean=model_loading_params['learn_mean'])

        elif model_loading_params['model'] == 'ffjord':
            model_single = load_ffjord_model(model_loading_params['ffjord_args'], model_loading_params['n_dim'],
                                             gaussian_params=model_loading_params['gaussian_params'],
                                             learn_mean=model_loading_params['learn_mean'])

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
        with torch.no_grad():
            for j, data in enumerate(loader):
                inp = data[0].float().to(device)
                labels = data[1].to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                Z.append(out.detach().cpu().numpy())
                tlabels.append(labels.detach().cpu().numpy())
        Z = np.concatenate(Z, axis=0).reshape(len(train_dataset), -1)
        tlabels = np.concatenate(tlabels, axis=0)

        # Learn SVC
        svc = make_pipeline(StandardScaler(), SVC(kernel='linear'))
        svc.fit(Z, tlabels)

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

        svc_score = svc.score(val_inZ, elabels)
        svc_scores.append(svc_score)

        print(f'Our approach ({i}) : {svc_score}')
        if svc_score >= best_score:
            best_score = svc_score
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
                                          learn_mean=model_loading_params['learn_mean'])

    elif model_loading_params['model'] == 'ffjord':
        model_single = load_ffjord_model(model_loading_params['ffjord_args'], model_loading_params['n_dim'],
                                         gaussian_params=model_loading_params['gaussian_params'],
                                         learn_mean=model_loading_params['learn_mean'])

    else:
        assert False, 'unknown model type!'

    model_single = WrappedModel(model_single)
    model_single.load_state_dict(torch.load(model_loading_params['loading_path'], map_location=device))
    torch.save(
        model_single.state_dict(), f"{save_dir}/best_classification_model.pth"
    )


if __name__ == '__main__':
    parser = testing_arguments()
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
        val_dataset = ImDataset(dataset_name=dataset_name, test=True)

    n_dim = dataset.X[0].shape[0]
    for sh in dataset.X[0].shape[1:]:
        n_dim *= sh
    if not dim_per_label:
        uni = np.unique(dataset.true_labels)
        dim_per_label = math.floor(n_dim / len(uni))

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
        gaussian_params = initialize_gaussian_params(dataset, eigval_list, isotrope='isotrope' in folder_name,
                                                     dim_per_label=dim_per_label, fixed_eigval=fixed_eigval)

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

    evaluate_classification(t_model_params, save_dir=save_dir, device=device)
