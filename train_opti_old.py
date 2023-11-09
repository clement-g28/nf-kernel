import ray.tune
from tqdm import tqdm
import numpy as np
import math
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

from utils.custom_glow import CGlow
from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model
from utils.training import training_arguments, seqflow_arguments, ffjord_arguments, cglow_arguments, moflow_arguments, \
    graphnvp_arguments, AddGaussianNoise
from utils.utils import write_dict_to_tensorboard, set_seed, create_folder, AverageMeter, \
    initialize_class_gaussian_params, \
    initialize_regression_gaussian_params, initialize_tmp_regression_gaussian_params

from utils.testing import project_between

from functools import partial
import os
import torch.nn.functional as F
from torch.utils.data import random_split
import torchvision
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from typing import Dict, Optional
from collections import defaultdict, deque
import numpy as np

from ray.tune import Stopper

from utils.utils import load_dataset


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


def init_model(var, noise, dataset, noise_x=None):
    fixed_eigval = None
    eigval_list = [var for i in range(dim_per_label)]

    if not dataset.is_regression_dataset():
        gaussian_params = initialize_class_gaussian_params(dataset, eigval_list, isotrope=args.isotrope_gaussian,
                                                           dim_per_label=dim_per_label, fixed_eigval=fixed_eigval)
    else:
        gaussian_params = initialize_regression_gaussian_params(dataset, eigval_list,
                                                                isotrope=args.isotrope_gaussian,
                                                                dim_per_label=dim_per_label,
                                                                fixed_eigval=fixed_eigval)

    folder_path = f'{args.dataset}/{args.model}/'
    if args.model == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        n_channel = dataset.n_channel
        model_single = CGlow(n_channel, args_cglow.n_flow, args_cglow.n_block, affine=args_cglow.affine,
                             conv_lu=not args_cglow.no_lu,
                             gaussian_params=gaussian_params, device=device, learn_mean=not args.fix_mean)
        folder_path += f'b{args_cglow.n_block}_f{args_cglow.n_flow}'
    elif args.model == 'seqflow':
        args_seqflow, _ = seqflow_arguments().parse_known_args()
        model_single = load_seqflow_model(dataset.in_size, args_seqflow.n_flow, gaussian_params=gaussian_params,
                                          learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var,
                                          dataset=dataset)
        folder_path += f'f{args_seqflow.n_flow}'
    elif args.model == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        # args_ffjord.n_block = args.n_block
        model_single = load_ffjord_model(args_ffjord, dataset.in_size, gaussian_params=gaussian_params,
                                         learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
        folder_path += f'b{args_ffjord.n_block}'
    elif args.model == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        args_moflow.noise_scale = noise
        args_moflow.noise_scale_x = noise_x
        model_single = load_moflow_model(args_moflow, gaussian_params=gaussian_params,
                                         learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
    else:
        assert False, 'unknown model type'

    return model_single


def train_opti(config):
    train_dataset = get_train_dataset()
    val_dataset = get_val_dataset()
    # Init model
    model = init_model(config["var"], config["noise"], train_dataset, config["noise_x"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    learnable_params = list(model.parameters())
    optimizer = optim.Adam(learnable_params, lr=config["lr"])

    beta = config["beta"]

    # TEST with weighted sampler
    train_loader = train_dataset.get_loader(args.batch_size, shuffle=True, drop_last=True, sampler=True)
    val_loader = get_val_dataset().get_loader(args.batch_size, shuffle=True, drop_last=True)
    loader_size = len(train_loader)

    for epoch in range(args.n_epoch):
        running_loss = 0.0
        running_logp = 0.0
        running_logdet = 0.0
        running_distloss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            model.downstream_process()
            itr = epoch * loader_size + i
            input, label = data

            input = train_dataset.format_data(input, device)
            label = label.to(device)

            log_p, distloss, logdet, o = model(input, label)

            nll_loss, log_p, log_det = train_dataset.format_loss(log_p, logdet)
            loss = nll_loss - beta * distloss

            loss = model.upstream_process(loss)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_logp += log_p.item()
            running_logdet += log_det.item()
            running_distloss += distloss.item()
            epoch_steps += 1
            if itr % 100 == 0:
                print("[%d, %5d] loss: %.3f, logP: %.3f, logdet %.3f, distloss %.3f" % (epoch + 1, i + 1,
                                                                                        running_loss / epoch_steps,
                                                                                        running_logp / epoch_steps,
                                                                                        running_logdet / epoch_steps,
                                                                                        running_distloss / epoch_steps))
                running_loss = 0.0

        # Evaluation
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        accuracy = 0
        val_logp = 0
        val_logdet = 0
        val_dist = 0
        model.eval()

        with torch.no_grad():
            for data in val_loader:
                input, label = data

                input = val_dataset.format_data(input, device)
                label = label.to(device)

                log_p, distloss, logdet, z = model(input, label)

                logdet = logdet.mean()

                nll_loss, log_p, log_det = val_dataset.format_loss(log_p, logdet)
                loss = nll_loss - beta * distloss
                val_loss += loss.cpu().numpy()
                val_steps += 1
                val_logp += log_p.cpu().numpy()
                val_logdet += log_det.cpu().numpy()
                val_dist += distloss.cpu().numpy()

                if dataset.is_regression_dataset():
                    # accuracy
                    means = model.means.detach().cpu().numpy()
                    np_z = z.detach().cpu().numpy()
                    np_label = label.detach().cpu().numpy()
                    proj, dot_val = project_between(np_z, means[0], means[1])
                    pred = dot_val.squeeze() * (model.label_max - model.label_min) + model.label_min
                    accuracy += np.power((pred - np_label), 2).mean()
                else:
                    accuracy += val_loss

        model.train()
        if epoch % args.save_each_epoch == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=(accuracy / val_steps), logp=(val_logp / val_steps),
                    logdet=(val_logdet / val_steps), distloss=(val_dist / val_steps))


if __name__ == "__main__":
    args, _ = training_arguments().parse_known_args()
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

    noise_str = ''
    if args.with_noise is not None:
        transform = [] if transform is None else transform
        transform += [AddGaussianNoise(0., args.with_noise)]
        noise_str = '_noise' + str(args.with_noise).replace('.', '')

    if transform is not None:
        transform = transforms.Compose(transform)

    # DATASET #
    dataset = load_dataset(args, args.dataset, args.model, transform=transform, add_feature=args.add_feature)

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

    n_dim = dataset.get_n_dim()

    # initialize gaussian params
    if not dataset.is_regression_dataset():
        if not args.dim_per_label:
            uni = np.unique(dataset.true_labels)
            dim_per_label = math.floor(n_dim / len(uni))
        else:
            dim_per_label = args.dim_per_label
    else:
        if not args.dim_per_label:
            dim_per_label = n_dim
        else:
            dim_per_label = args.dim_per_label

    from datetime import datetime

    date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

    folder_path = f'./checkpoint/{args.dataset}/{args.model}/ray_idx/{date}'
    create_folder(folder_path)

    path = f'{folder_path}/train_idx'
    train_dset.save_split(path)
    if val_dset is not None:
        path = f'{folder_path}/val_idx'
        val_dset.save_split(path)

    config = {
        "var": tune.uniform(10.0, 20.0),
        "beta": tune.randint(10, 200),
        "noise": tune.uniform(0.2, 0.9),
        "noise_x": tune.uniform(0.05, 0.3),
        "lr": tune.loguniform(1e-4, 0.001),
        "batch_size": tune.choice([150, 200, 250])
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="min",
        max_t=args.n_epoch,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["var", "beta", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "logp", "logdet", "distloss", "training_iteration"])

    train_dataset_id = ray.put(train_dset)
    val_dataset_id = ray.put(val_dset)
    result = tune.run(
        partial(train_opti),
        resources_per_trial={"cpu": 4, "gpu": 0.25},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter,
        stop=nan_stopper)

    best_trial = result.get_best_trial("accuracy", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))