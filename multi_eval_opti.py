from evaluate_opti import main as evaluate_model
import torch
import os
from argparse import Namespace
from utils.testing import learn_or_load_modelhyperparams, load_split_dataset, testing_arguments, \
    initialize_gaussian_params


def process_eval(args):
    evaluate_model(args)
    torch.cuda.empty_cache()


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
