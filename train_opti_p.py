import ray.tune
import math
import torch
from torch import nn, optim
from torchvision import transforms

from utils.custom_glow import CGlow
from utils.models import load_seqflow_model, load_cglow_model, load_ffjord_model, load_moflow_model
from utils.training import training_arguments, seqflow_arguments, ffjord_arguments, cglow_arguments, moflow_arguments, \
    graphnvp_arguments, AddGaussianNoise
from utils.utils import write_dict_to_tensorboard, set_seed, create_folder, AverageMeter, \
    initialize_class_gaussian_params, initialize_regression_gaussian_params

from utils.testing import project_between

from functools import partial
import os
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from typing import Dict
import numpy as np

from ray.tune import Stopper

from utils.utils import load_dataset

from utils.dataset import GraphDataset

from evaluate import classification_score, learn_or_load_modelhyperparams
from sklearn.svm import SVC
from sklearn.linear_model import Ridge

from train import train


def nan_stopper(trial_id: str, result: Dict):
    metric_result = result.get('accuracy')
    return math.isnan(metric_result) or math.isinf(metric_result)


class NaNStopper(Stopper):
    """Early stop single trials when NaN or inf

    Args:
        metric: Metric to check for NaN or inf
    """

    def __init__(self, metric: str):
        super().__init__()
        self._metric = metric

    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)

        return math.isnan(metric_result) or math.isinf(metric_result)

    def stop_all(self):
        return False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = None
train_dataset_id = None
val_dataset_id = None
dim_per_label = None


def get_train_dataset():
    return ray.get(train_dataset_id)


def get_val_dataset():
    return ray.get(val_dataset_id)


def train_opti(config):
    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()

    trial_dir = tune.get_trial_dir()

    train(args, config, train_dataset, val_dataset, trial_dir, is_raytuning=True)


if __name__ == "__main__":
    # args, _ = training_arguments().parse_known_args()
    parser = training_arguments()
    args, _ = parser.parse_known_args()
    print(args)

    # set_seed(0)

    if args.dataset == 'mnist':
        transform = [
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    elif args.dataset == 'cifar10':
        transform = [
            # transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    elif args.dataset == 'olivetti_faces':
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    else:
        transform = None

    if args.with_noise is not None:
        transform = [] if transform is None else transform
        transform += [AddGaussianNoise(0., args.with_noise)]

    if transform is not None:
        transform = transforms.Compose(transform)

    # DATASET #
    # dataset = load_dataset(args, args.dataset, args.model, transform=transform, add_feature=args.add_feature)
    dataset = load_dataset(args, args.dataset, args.model, transform=transform)  # do not give the add_feature here !

    if args.reduce_class_size:
        dataset.reduce_dataset('every_class', how_many=args.reduce_class_size)

    if args.unique_label:
        dataset.reduce_dataset('one_class', label=args.unique_label)
        dataset.true_labels[:] = 0
    elif args.multi_label is not None:
        labels = list(map(int, args.multi_label.strip('[]').split(',')))
        dataset.reduce_dataset('multi_class', label=labels)

    validation = args.validation
    if validation > 0:
        # TEST with stratified sample
        train_dset, val_dset = dataset.split_dataset(validation, stratified=True)
        train_dset.ori_X = train_dset.X
        train_dset.ori_true_labels = train_dset.true_labels
        val_dset.ori_X = val_dset.X
        val_dset.ori_true_labels = val_dset.true_labels
    else:
        train_dset = dataset
        val_dset = None

    device = 'cuda:0'

    from datetime import datetime

    date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    folder_path = f'./checkpoint/{args.dataset}/{args.model}/ray_idx/{date}'
    create_folder(folder_path)

    path = f'{folder_path}/train_idx'
    train_dset.save_split(path)
    if val_dset is not None:
        path = f'{folder_path}/val_idx'
        val_dset.save_split(path)

    # Config MUTAG
    # config = {
    #     "var": ('uniform', tune.uniform(1.1, 1.4)),
    #     "beta": tune.randint(10, 200),
    #     "noise": tune.uniform(0.1, 0.5),
    #     # "noise_x": tune.uniform(0.05, 0.3),
    #     "noise_x": None,
    #     "lr": tune.loguniform(1e-4, 0.0004),
    #     "batch_size": tune.choice([10]),
    #     "add_feature": tune.randint(0, 20),
    #     "split_graph_dim": True
    # }
    # Config Letter-med
    # config = {
    #     "var": ('uniform', tune.uniform(10, 30)),
    #     "beta": tune.randint(10, 200),
    #     "noise": tune.uniform(0.2, 0.6),
    #     "noise_x": tune.uniform(0.05, 0.3),
    #     # "noise_x": None,
    #     "lr": tune.loguniform(1e-3, 0.01),
    #     "batch_size": tune.choice([50, 100, 150, 200, 250]),
    #     "add_feature": tune.randint(0, 20),
    #     "split_graph_dim": True
    # }
    config = {
        "var_type": 'uniform',
        "var": tune.uniform(0.1, 0.5),
        "beta": tune.randint(10, 200),
        "noise": tune.uniform(0.3, 0.6),
        # "noise_x": tune.uniform(0.05, 0.3),
        "noise_x": None,
        "lr": tune.loguniform(1e-4, 0.001),
        "batch_size": tune.choice([100, 150, 200]),
        "add_feature": tune.randint(0, 20),
        "split_graph_dim": True
    }

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="min",
        # mode="max",
        max_t=args.n_epoch,
        grace_period=10,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["var", "beta", "lr", "batch_size", "add_feature"],
        metric_columns=["loss", "nll", "logp", "logdet", "distloss", "training_iteration"])

    train_dataset_id = ray.put(train_dset)
    val_dataset_id = ray.put(val_dset)
    result = tune.run(
        partial(train_opti),
        resources_per_trial={"cpu": 4, "gpu": 0.5},
        config=config,
        num_samples=50,
        scheduler=scheduler,
        progress_reporter=reporter,
        stop=nan_stopper)

    best_trial = result.get_best_trial("accuracy", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
