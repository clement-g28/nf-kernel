import numpy as np
import math
import os
import torch

from utils.utils import set_seed, create_folder, initialize_class_gaussian_params, \
    initialize_regression_gaussian_params, load_dataset

from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model, load_cglow_model
from utils.dataset import GraphDataset, GRAPH_CLASSIFICATION_DATASETS
from utils.testing import learn_or_load_modelhyperparams, load_split_dataset, testing_arguments, \
    initialize_gaussian_params
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments

from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from evaluate import evaluate_regression, create_figures_XZ, evaluate_preimage, \
    evaluate_preimage2, evaluate_distances, evaluate_classification, generate_meanclasses, \
    test_generation_on_eigvec, evaluate_projection_1model, evaluate_graph_interpolations, evaluate_image_interpolations, \
    evaluate_p_value, compression_evaluation, create_figure_train_projections


def main(args):
    print(args)
    if args.seed is not None:
        set_seed(args.seed)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]

    is_last_chckpt = False
    flow_path = f'{args.folder}/save/best_{args.model_to_use}_model.pth'
    # assert os.path.exists(flow_path), f'snapshot path {flow_path} does not exists'
    if not os.path.exists(flow_path):
        import glob
        # os.chdir(args.folder)
        saves = []
        for file in glob.glob(f"{args.folder}/checkpoint_*/checkpoint"):
            saves.append(file)

        saves.sort()
        print(saves)
        flow_path = saves[-1]
        is_last_chckpt = True
        print('selecting last saved checkpoint as model.')
    # folder_name = args.folder.split('/')[-1]

    # splits = folder_name.split(',')
    # for split in splits:
    #     if 'var' in split:
    #         var = float(split.split('_')[0].split('var=')[-1])
    #

    import json

    params_path = args.folder + '/params.json'
    # Opening JSON file
    with open(params_path) as json_file:
        data = json.load(json_file)
    var = data['var']
    add_feature = data['add_feature']
    batch_size = data['batch_size']
    split_graph_dim = data['split_graph_dim']

    # DATASET #
    dataset = load_dataset(args, dataset_name, model_type, to_evaluate=True, add_feature=add_feature)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Retrieve parameters from name
    mean_of_eigval = var
    fixed_eigval = None
    n_block = 2  # base value
    n_flow = 32  # base value
    reg_use_var = False

    # Load the splits of the dataset used in the training phase
    train_idx_path = f'{args.folder}/train_idx.npy'
    val_idx_path = f'{args.folder}/val_idx.npy'
    train_dataset, val_dataset = load_split_dataset(dataset, train_idx_path, val_idx_path,
                                                    reselect_val_idx=args.reselect_val_idx)

    # reduce train dataset size (fitting too long)
    # print('Train dataset reduced in order to accelerate. (stratified)')
    # train_dataset.reduce_dataset_ratio(0.1, stratified=True)
    # val_dataset.reduce_regression_dataset(0.1, stratified=True)

    n_dim = dataset.get_n_dim()

    if not dataset.is_regression_dataset():
        uni = np.unique(dataset.true_labels)
        dim_per_label = math.floor(n_dim / len(uni))
    else:
        dim_per_label = n_dim

    # initialize gaussian params
    eigval_list = [mean_of_eigval for i in range(dim_per_label)]

    # print('split graph dim should be set to set manually in evaluate opti, default to False...')
    # split_graph_dim = False
    if not dataset.is_regression_dataset():
        gaussian_params = initialize_class_gaussian_params(dataset, eigval_list, isotrope=False,
                                                           dim_per_label=dim_per_label, fixed_eigval=fixed_eigval,
                                                           split_graph_dim=split_graph_dim)
    else:
        gaussian_params = initialize_regression_gaussian_params(dataset, eigval_list,
                                                                isotrope=True,
                                                                dim_per_label=dim_per_label,
                                                                fixed_eigval=fixed_eigval)

    learn_mean = True
    if model_type == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        args_cglow.n_flow = n_flow
        n_channel = dataset.n_channel
        model_single = load_cglow_model(args_cglow, n_channel, gaussian_params=gaussian_params,
                                        learn_mean=learn_mean, device=device)
    elif model_type == 'seqflow':
        model_single = load_seqflow_model(n_dim, n_flow, gaussian_params=gaussian_params,
                                          learn_mean=learn_mean, reg_use_var=reg_use_var, dataset=dataset)

    elif model_type == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        args_ffjord.n_block = n_block
        model_single = load_ffjord_model(args_ffjord, n_dim, gaussian_params=gaussian_params,
                                         learn_mean=learn_mean, reg_use_var=reg_use_var, dataset=dataset)
    elif model_type == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        model_single = load_moflow_model(args_moflow,
                                         gaussian_params=gaussian_params,
                                         learn_mean=learn_mean, reg_use_var=reg_use_var,
                                         dataset=dataset)
    else:
        assert False, 'unknown model type!'

    if not is_last_chckpt:
        model_state = torch.load(flow_path, map_location=device)
    else:
        model_state, optimizer_state = torch.load(flow_path, map_location=device)
    model_single.load_state_dict(model_state)
    model = model_single

    model = model.to(device)
    model.eval()

    os.chdir(args.folder)

    save_dir = './save'
    create_folder(save_dir)

    # generate flows
    # std_noise = .1 if dataset_name in ['double_moon', 'single_moon', 'swissroll'] else None
    # # fig_limits = ((-23,23),(-4,4))
    # fig_limits = None
    # model.interpret_transformation(train_dataset, save_dir, device, std_noise=std_noise, fig_limits=fig_limits)

    eval_type = args.eval_type
    if eval_type == 'classification':
        dataset_name_eval = ['mnist', 'cifar10', 'double_moon', 'iris', 'bcancer'] + GRAPH_CLASSIFICATION_DATASETS
        assert dataset_name in dataset_name_eval, f'Classification can only be evaluated on {dataset_name_eval}'
        # predmodel = evaluate_classification(model, train_dataset, val_dataset, save_dir, device)
        evaluate_classification(model, train_dataset, val_dataset, save_dir, device)
        # _, Z = create_figures_XZ(model, train_dataset, save_dir, device, std_noise=0.1,
        #                          only_Z=isinstance(dataset, GraphDataset))
        # print_as_mol = True
        # print_as_graph = False
        # refresh_means = False
        # print(f'(print_as_mol, print_as_graph, refresh_means) are set manually to '
        #       f'({print_as_mol},{print_as_graph},{refresh_means}).')
        # computed_means = model.refresh_classification_mean_classes(Z, train_dataset.true_labels) if refresh_means \
        #     else None
        # evaluate_preimage(model, val_dataset, device, save_dir, print_as_mol=print_as_mol,
        #                   print_as_graph=print_as_graph, eval_type=eval_type, means=computed_means)
        # evaluate_preimage2(model, val_dataset, device, save_dir, n_y=20, n_samples_by_y=10, print_as_mol=print_as_mol,
        #                    print_as_graph=print_as_graph, eval_type=eval_type, predmodel=predmodel,
        #                    means=computed_means)
        # if isinstance(val_dataset, GraphDataset):
        #     evaluate_graph_interpolations(model, val_dataset, device, save_dir, n_sample=9, n_interpolation=30, Z=Z,
        #                                   print_as_mol=print_as_mol, print_as_graph=print_as_graph, eval_type=eval_type,
        #                                   label=None)
    elif eval_type == 'generation':
        dataset_name_eval = ['mnist', 'cifar10', 'olivetti_faces']
        assert dataset_name in dataset_name_eval, f'Generation can only be evaluated on {dataset_name_eval}'
        # GENERATION
        create_folder(f'{save_dir}/test_generation')

        img_size = dataset.in_size
        z_shape = model.calc_last_z_shape(img_size)
        from utils.training import AddGaussianNoise
        from torchvision import transforms
        dataset.transform = transforms.Compose(dataset.transform.transforms + [AddGaussianNoise(0., .2)])
        print('Gaussian noise added to transforms...')
        evaluate_image_interpolations(model, dataset, device, save_dir, n_sample=20, n_interpolation=10, label=None)

        # how_much = [1, 10, 30, 50, 78]
        # how_much = [dim_per_label, n_dim]
        how_much = [1, dim_per_label]
        print(f'how_much is set manually to {how_much}.')
        for n in how_much:
            test_generation_on_eigvec(model, val_dataset, gaussian_params=gaussian_params, z_shape=z_shape,
                                      how_much_dim=n, device=device, sample_per_label=10, save_dir=save_dir)
        generate_meanclasses(model, train_dataset, device, save_dir)
    elif eval_type == 'projection':
        img_size = dataset.in_size
        z_shape = model.calc_last_z_shape(img_size)
        # PROJECTIONS
        proj_type = 'gp'
        batch_size = 100
        eval_gaussian_std = .1
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
        elif dataset_name in ['single_moon', 'double_moon']:
            noise_type = 'gaussian'
            std_noise = .1
            create_figure_train_projections(model, train_dataset, std_noise=std_noise, save_path=save_dir,
                                            device=device)
            n_principal_dim = np.count_nonzero(gaussian_params[0][-2] > 1)
            # evaluate distance n times to calculate the p-value
            n_times = 20
            kpca_types = ['linear', 'rbf', 'poly', 'sigmoid']
            proj_type = 'gp'
            print(f'(n_times, kpca_types, proj_type) are set manually to '
                  f'({n_times},{kpca_types},{proj_type}).')
            distance_results = {ktype + '-PCA': [] for ktype in kpca_types}
            distance_results['Our-' + proj_type] = []
            for n in range(n_times):
                res = evaluate_distances(model, train_dataset, val_dataset, gaussian_params=gaussian_params,
                                         z_shape=z_shape, how_much=n_principal_dim,
                                         kpca_types=kpca_types,
                                         device=device, save_dir=save_dir, proj_type=proj_type, noise_type=noise_type,
                                         eval_gaussian_std=std_noise)
                for ktype in kpca_types:
                    distance_results[ktype + '-PCA'].append(res[ktype + '-PCA'])
                distance_results['Our-' + proj_type].append(res['Our-' + proj_type])
            print(distance_results)
            # p-value
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
        else:
            dataset_name_eval = ['mnist', 'single_moon', 'double_moon']
            assert dataset_name in dataset_name_eval, f'Projection can only be evaluated on {dataset_name_eval}'
    elif eval_type == 'regression':
        assert dataset.is_regression_dataset(), 'the dataset is not made for regression purposes'
        predmodel = evaluate_regression(model, train_dataset, val_dataset, save_dir, device)
        _, Z = create_figures_XZ(model, train_dataset, save_dir, device, std_noise=0.1,
                                 only_Z=isinstance(dataset, GraphDataset))
        print_as_mol = True
        print_as_graph = True
        print(f'(print_as_mol, print_as_graph) are set manually to '
              f'({print_as_mol},{print_as_graph}).')
        # evaluate_preimage(model, val_dataset, device, save_dir, print_as_mol=print_as_mol,
        #                   print_as_graph=print_as_graph)
        evaluate_preimage2(model, val_dataset, device, save_dir, n_y=24, n_samples_by_y=1,
                           print_as_mol=print_as_mol, print_as_graph=print_as_graph, predmodel=predmodel, debug=True)
        if isinstance(val_dataset, GraphDataset):
            evaluate_graph_interpolations(model, val_dataset, device, save_dir, n_sample=100, n_interpolation=30, Z=Z,
                                          print_as_mol=print_as_mol, print_as_graph=print_as_graph)

    del dataset
    del val_dataset
    del train_dataset
    model.del_model_from_gpu()


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
    # parser.add_argument("--add_feature", type=int, default=None)
    args = parser.parse_args()
    args.seed = 3
    main(args)
