import os
import torch
import json
import glob

from utils.utils import set_seed, create_folder, load_dataset

from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model, load_cglow_model
from utils.testing import load_split_dataset, testing_arguments, initialize_gaussian_params
from utils.training import ffjord_arguments, cglow_arguments, moflow_arguments, seqflow_arguments

from get_best_opti import retrieve_config


def main(args):
    print(args)
    if args.seed is not None:
        set_seed(args.seed)

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]

    params_path = args.folder + '/params.json'
    config, batch_size = retrieve_config(params_path)

    # DATASET #
    dataset = load_dataset(args, dataset_name, model_type, to_evaluate=True, add_feature=config['add_feature'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_last_chckpt = False
    flow_path = f'{args.folder}/save/best_{args.model_to_use}_model.pth'
    # assert os.path.exists(flow_path), f'snapshot path {flow_path} does not exists'
    if not os.path.exists(flow_path):
        # os.chdir(args.folder)
        saves = []
        for file in glob.glob(f"{args.folder}/checkpoint_*/checkpoint"):
            saves.append(file)

        saves.sort()
        print(saves)
        flow_path = saves[-1]
        is_last_chckpt = True
        print('selecting last saved checkpoint as model.')

    # Load the splits of the dataset used in the training phase
    train_idx_path = f'{args.folder}/train_idx.npy'
    test_idx_path = f'{args.folder}/test_idx.npy'
    train_dataset, test_dataset = load_split_dataset(dataset, train_idx_path, test_idx_path,
                                                     reselect_val_idx=args.reselect_val_idx, split_type='test')

    # reduce train dataset size (fitting too long)
    if args.reduce_test_dataset_size is not None:
        train_dataset = train_dataset.duplicate()
        print('Train dataset reduced in order to accelerate. (stratified)')
        train_dataset.reduce_dataset_ratio(args.reduce_test_dataset_size, stratified=True)

    n_dim, dim_per_label = dataset.get_dim_per_label(return_total_dim=True)
    config['dim_per_label'] = dim_per_label

    # initialize gaussian params
    gaussian_params = initialize_gaussian_params(args, dataset, config['var_type'], config['mean_of_eigval'],
                                                 config['dim_per_label'], dataset.is_regression_dataset(),
                                                 config['split_graph_dim'], fixed_eigval=config['fixed_eigval'],
                                                 gaussian_eigval=config['gaussian_eigval'],
                                                 add_feature=config['add_feature'])

    # n_flow or n_block should be specified in arguments as there are not saved during training
    learn_mean = True
    reg_use_var = False
    if model_type == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        # args_cglow.n_flow = n_flow
        n_channel = dataset.n_channel
        model_single = load_cglow_model(args_cglow, n_channel, gaussian_params=gaussian_params,
                                        learn_mean=learn_mean, device=device)
    elif model_type == 'seqflow':
        args_seqflow, _ = seqflow_arguments().parse_known_args()
        model_single = load_seqflow_model(n_dim, args_seqflow.n_flow, gaussian_params=gaussian_params,
                                          learn_mean=learn_mean, reg_use_var=reg_use_var, dataset=dataset)
    elif model_type == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        # args_ffjord.n_block = n_block
        model_single = load_ffjord_model(args_ffjord, n_dim, gaussian_params=gaussian_params,
                                         learn_mean=learn_mean, reg_use_var=reg_use_var, dataset=dataset)
    elif model_type == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        model_single = load_moflow_model(args_moflow,
                                         gaussian_params=gaussian_params,
                                         learn_mean=learn_mean, reg_use_var=reg_use_var,
                                         dataset=dataset, add_feature=config['add_feature'])
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

    from evaluate import launch_evaluation

    eval_type = args.eval_type
    launch_evaluation(eval_type, dataset_name, model, gaussian_params, train_dataset, test_dataset, save_dir, device,
                      batch_size, args.n_permutation_test)

    del dataset
    del test_dataset
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
