from evaluate_opti import main as evaluate_model
import torch
import os
from argparse import Namespace
from utils.testing import learn_or_load_modelhyperparams, load_split_dataset, testing_arguments, \
    initialize_gaussian_params

import torch
import gc


def process_eval(args):
    evaluate_model(args)
    # torch.cuda.empty_cache()

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    # # prints currently alive Tensors and Variables
    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
    #             print(type(obj), obj.size())
    #     except:
    #         pass


if __name__ == "__main__":
    choices = ['classification', 'projection', 'generation', 'regression']
    best_model_choices = ['classification', 'projection', 'regression']
    for choice in best_model_choices.copy():
        best_model_choices.append(choice + '_train')
    best_model_choices += [None]
    parser = testing_arguments()
    parser.add_argument('--eval_type', type=str, default='classification', choices=choices, help='evaluation type')
    parser.add_argument('--model_to_use', type=str, default=None, choices=best_model_choices,
                        help='what best model to use for the evaluation')
    parser.add_argument("--method", default=0, type=int, help='select between [0,1,2]')
    # parser.add_argument("--add_feature", type=int, default=None)
    args = parser.parse_args()
    if args.model_to_use is None:
        if args.eval_type != 'generation':
            args.model_to_use = args.eval_type
        else:
            assert False, f'the model_to_use argument should be defined if you want to evaluate the model for ' \
                          f'generation, it can be {best_model_choices}'
    args.seed = 3

    # evaluate
    current_folder = os.getcwd()

    arguments_test = []
    for file in os.listdir(f"{args.folder}"):
        if os.path.isdir(f"{args.folder}/{file}") and "train_opti" in file:
            has_checkpoint = False
            for file1 in os.listdir(f"{args.folder}/{file}"):
                if os.path.isdir(f"{args.folder}/{file}/{file1}") and 'checkpoint' in file1:
                    has_checkpoint = True
                    break
            if has_checkpoint:
                args_copy = Namespace(**vars(args))
                args_copy.folder = f"{args.folder}/{file}"
                arguments_test.append(args_copy)

    for i in range(len(arguments_test)):
        process_eval(arguments_test[i])
        os.chdir(current_folder)
