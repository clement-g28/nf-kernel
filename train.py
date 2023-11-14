import time

from tqdm import tqdm
import numpy as np
import math
import torch
import os
from ray import tune
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils.models import load_seqflow_model, load_ffjord_model, load_moflow_model, load_graphnvp_model, \
    load_cglow_model
from utils.training import training_arguments, ffjord_arguments, seqflow_arguments, cglow_arguments, moflow_arguments, \
    graphnvp_arguments, AddGaussianNoise
from utils.utils import write_dict_to_tensorboard, set_seed, create_folder, AverageMeter, \
    initialize_class_gaussian_params, initialize_regression_gaussian_params, initialize_tmp_regression_gaussian_params

from utils.utils import load_dataset
from utils.dataset import GraphDataset
from utils.testing import project_between, initialize_gaussian_params

from evaluate import classification_score, learn_or_load_modelhyperparams
from sklearn.svm import SVC
from sklearn.linear_model import Ridge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def train(args, model_single, add_path, train_dataset, val_dataset=None):
#     model = nn.DataParallel(model_single)
#     model = model.to(device)
#
#     learnable_params = list(model.parameters())
#     optimizer = optim.Adam(learnable_params, lr=args.lr)
#
#     save_dir = f"./checkpoint/{add_path}"
#     create_folder(save_dir)
#     create_folder(f'{save_dir}/sample')
#
#     if args.use_tb:
#         writer = SummaryWriter(save_dir)
#
#     beta = args.beta
#     # z_shape = model_single.calc_last_z_shape(dataset.im_size)
#
#     # TEST with weighted sampler
#     train_loader = train_dataset.get_loader(args.batch_size, shuffle=True, drop_last=True, sampler=True)
#     if val_dataset is not None:
#         val_loader = val_dataset.get_loader(args.batch_size, shuffle=True, drop_last=False)
#     loader_size = len(train_loader)
#
#     with tqdm(range(args.n_epoch)) as pbar:
#         for epoch in pbar:
#             for i, data in enumerate(train_loader):
#                 optimizer.zero_grad()
#                 model_single.downstream_process()
#                 itr = epoch * loader_size + i
#                 input, label = data
#
#                 input = train_dataset.format_data(input, device)
#                 label = label.to(device)
#
#                 if itr == 0:
#                     with torch.no_grad():
#                         log_p, distloss, logdet, o = model.module(input, label)
#
#                         continue
#
#                 else:
#                     log_p, distloss, logdet, o = model(input, label)
#
#                 # logdet = logdet.mean()
#
#                 # nll_loss, log_p, log_det = calc_loss(log_p, logdet, dataset.im_size, n_bins)
#                 nll_loss, log_p, log_det = train_dataset.format_loss(log_p, logdet)
#                 loss = nll_loss - beta * distloss
#
#                 loss = model_single.upstream_process(loss)
#                 # loss.clamp_(-10000,10000)
#
#                 model.zero_grad()
#                 # loss.backward()
#
#                 # if epoch < 100:
#                 #     loss += log_det
#                 # loss.backward(retain_graph=True)
#                 # loss2 = torch.pow(log_det, 2)
#                 # loss2.backward()
#
#                 loss.backward()
#
#                 # Gradient clipping
#                 # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
#
#                 # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
#                 warmup_lr = args.lr
#                 optimizer.param_groups[0]["lr"] = warmup_lr
#                 optimizer.step()
#
#                 pbar.set_description(
#                     f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}; distloss: {distloss.item():.7f}"
#                 )
#
#                 if itr % args.write_train_loss_every == 0:
#                     if args.use_tb:
#                         loss_dict = {"tot_loss": loss.item(),
#                                      "logP": log_p.item(),
#                                      "logdet": log_det.item(),
#                                      "distloss": distloss.item()}
#                         write_dict_to_tensorboard(writer, loss_dict, base_name="TRAIN", iteration=itr)
#
#                         means_dict = {}
#                         for i, mean in enumerate(model_single.means):
#                             m = np.mean(mean.detach().cpu().numpy())
#                             means_dict[f'mean{i}'] = m.item()
#                         write_dict_to_tensorboard(writer, means_dict, base_name=f"MEANS", iteration=itr)
#
#                 if itr % args.sample_every == 0:
#                     with torch.no_grad():
#                         model.eval()
#                         model_single.sample_evaluation(itr, train_dataset, val_dataset, save_dir,
#                                                        writer=writer)
#                         model.train()
#
#                 del loss
#                 torch.cuda.empty_cache()
#
#             # Evaluation
#             if val_dataset is not None:
#                 model.eval()
#
#                 valLossMeter = AverageMeter()
#                 valLogPMeter = AverageMeter()
#                 valLogDetMeter = AverageMeter()
#                 valDistLossMeter = AverageMeter()
#
#                 with torch.no_grad():
#                     for data in val_loader:
#                         input, label = data
#
#                         input = val_dataset.format_data(input, device)
#                         label = label.to(device)
#
#                         log_p, distloss, logdet, z = model(input, label)
#
#                         logdet = logdet.mean()
#
#                         # nll_loss, log_p, log_det = calc_loss(log_p, logdet, dataset.im_size, n_bins)
#                         nll_loss, log_p, log_det = val_dataset.format_loss(log_p, logdet)
#                         loss = nll_loss - beta * distloss
#
#                         valLossMeter.update(loss.item())
#                         valLogPMeter.update(log_p.item())
#                         valLogDetMeter.update(log_det.item())
#                         valDistLossMeter.update(distloss.item())
#
#                     if args.use_tb:
#                         loss_dict = {"tot_loss": valLossMeter.avg,
#                                      "logP": valLogPMeter.avg,
#                                      "logdet": valLogDetMeter.avg,
#                                      "distloss": distloss.item()}
#                         write_dict_to_tensorboard(writer, loss_dict, base_name="VAL", iteration=itr)
#
#                     with torch.no_grad():
#                         model_single.evaluate(itr, input, label, z, val_dataset, save_dir, writer=writer)
#                         # images = model_single.reverse(z).cpu().data
#                         # images = train_dataset.rescale(images)
#                         # utils.save_image(
#                         #     images,
#                         #     f"{save_dir}/sample/val_{str(itr + 1).zfill(6)}.png",
#                         #     normalize=True,
#                         #     nrow=10,
#                         #     range=(0, 255),
#                         # )
#                         # img_grid = utils.make_grid(images, nrow=10, padding=2, pad_value=0, normalize=True,
#                         #                            range=(0, 255), scale_each=False)
#                         # writer.add_image('val', img_grid)
#                 model.train()
#
#             if epoch > args.save_at_epoch and epoch % args.save_each_epoch == 0:
#                 torch.save(model.state_dict(), f"{save_dir}/model_{str(epoch).zfill(6)}.pt")


def prepare_model(args, var, noise, dataset, noise_x=None, add_feature=None, split_graph_dim=False):
    n_dim = dataset.get_n_dim(add_feature=add_feature)

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

    fixed_eigval = None
    gaussian_eigval = None
    var_type = var[0]  # 'manual', 'uniform' or 'gaussian'
    var_params = var[1]
    gaussian_params = initialize_gaussian_params(args, dataset, var_type, var_params, dim_per_label,
                                                 args.isotrope_gaussian, split_graph_dim, fixed_eigval,
                                                 gaussian_eigval, add_feature)

    if args.model == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        n_channel = dataset.n_channel
        model_single = load_cglow_model(args_cglow, n_channel, gaussian_params=gaussian_params,
                                        learn_mean=not args.fix_mean, device=device)
    elif args.model == 'seqflow':
        args_seqflow, _ = seqflow_arguments().parse_known_args()
        model_single = load_seqflow_model(n_dim, args_seqflow.n_flow, gaussian_params=gaussian_params,
                                          learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var,
                                          dataset=dataset)
    elif args.model == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        # args_ffjord.n_block = args.n_block
        model_single = load_ffjord_model(args_ffjord, n_dim, gaussian_params=gaussian_params,
                                         learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
    elif args.model == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        args_moflow.noise_scale = noise
        args_moflow.noise_scale_x = noise_x
        model_single = load_moflow_model(args_moflow, gaussian_params=gaussian_params,
                                         learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset,
                                         add_feature=add_feature)
    else:
        assert False, 'unknown model type'

    return model_single


def validate_model(val_loader, val_dataset, model, beta, config, device):
    val_loss = 0.0
    val_steps = 0
    accuracy = 0
    val_nll = 0
    val_logp = 0
    val_logdet = 0
    val_dist = 0
    for data in val_loader:
        input, label = data

        input = val_dataset.format_data(input, device, add_feature=config["add_feature"])
        label = label.to(device)

        log_p, distloss, logdet, z = model(input, label)

        logdet = logdet.mean()

        nll_loss, log_p, log_det = val_dataset.format_loss(log_p, logdet)
        loss = nll_loss - beta * distloss
        val_loss += loss.cpu().numpy()
        val_steps += 1
        val_nll += nll_loss.cpu().numpy()
        val_logp += log_p.cpu().numpy()
        val_logdet += log_det.cpu().numpy()
        val_dist += distloss.cpu().numpy()

        if val_dataset.is_regression_dataset():
            # accuracy
            means = model.means.detach().cpu().numpy()
            np_z = z.detach().cpu().numpy()
            np_label = label.detach().cpu().numpy()
            proj, dot_val = project_between(np_z, means[0], means[1])
            pred = dot_val.squeeze() * (model.label_max - model.label_min) + model.label_min
            accuracy += np.power((pred - np_label), 2).mean()
        else:
            accuracy += val_loss
            # accuracy += (val_logp - val_nll)

    return val_loss, val_steps, accuracy, val_nll, val_logp, val_logdet, val_dist


def train(args, config, train_dataset, val_dataset, test_dataset, save_dir, is_raytuning):
    path = os.path.join(save_dir, "train_idx")
    train_dataset.save_split(path)
    if val_dataset is not None:
        path = os.path.join(save_dir, "val_idx")
        val_dataset.save_split(path)
    if test_dataset is not None:
        path = os.path.join(save_dir, "test_idx")
        test_dataset.save_split(path)

    if not is_raytuning and args.use_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(save_dir)

    # Init model
    model = prepare_model(args, (config["var_type"], config["var"]), config["noise"], train_dataset, config["noise_x"],
                          config["add_feature"], config["split_graph_dim"])

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
    train_loader = train_dataset.get_loader(config['batch_size'], shuffle=True, drop_last=False, sampler=True)
    val_loader = val_dataset.get_loader(config['batch_size'], shuffle=True, drop_last=False)
    # loader_size = len(train_loader)

    best_accuracy = math.inf
    # best_accuracy = -math.inf

    for epoch in range(args.n_epoch):
        running_loss = 0.0
        running_nll = 0.0
        running_logp = 0.0
        running_logdet = 0.0
        running_distloss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            model.downstream_process()
            # itr = epoch * loader_size + i
            input, label = data

            input = train_dataset.format_data(input, device, add_feature=config["add_feature"])
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
            running_nll += nll_loss.item()
            running_logp += log_p.item()
            running_logdet += log_det.item()
            running_distloss += distloss.item()
            epoch_steps += 1

        print("Epoch: %d, loss: %.3f, NLL: %.3f, logP: %.3f, logdet %.3f, distloss %.3f" % (epoch + 1,
                                                                                            running_loss / epoch_steps,
                                                                                            running_nll / epoch_steps,
                                                                                            running_logp / epoch_steps,
                                                                                            running_logdet / epoch_steps,
                                                                                            running_distloss / epoch_steps))

        if not is_raytuning and args.use_tb:
            loss_dict = {"tot_loss": (running_loss / epoch_steps),
                         "NLL": (running_nll / epoch_steps),
                         "logP": (running_logp / epoch_steps),
                         "logdet": (running_logdet / epoch_steps),
                         "distloss": (running_distloss / epoch_steps)}
            write_dict_to_tensorboard(writer, loss_dict, base_name="TRAIN", iteration=epoch)

            means_dict = {}
            for i, mean in enumerate(model.means):
                m = np.mean(mean.detach().cpu().numpy())
                means_dict[f'mean{i}'] = m.item()
            write_dict_to_tensorboard(writer, means_dict, base_name=f"MEANS", iteration=epoch)

        # Evaluation
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        accuracy = 0
        val_nll = 0
        val_logp = 0
        val_logdet = 0
        val_dist = 0
        model.eval()

        with torch.no_grad():
            if isinstance(train_dataset, GraphDataset):  # Permutations for Graphs
                n_permutation = 10
                for n in range(n_permutation):
                    results = validate_model(val_loader, val_dataset, model, beta, config, device)
                    # val_loss, val_steps, accuracy, val_nll, val_logp, val_logdet, val_dist
                    val_loss += results[0]
                    val_steps += results[1]
                    accuracy += results[2]
                    val_nll += results[3]
                    val_logp += results[4]
                    val_logdet += results[5]
                    val_dist += results[6]
            else:
                val_loss, val_steps, accuracy, \
                val_nll, val_logp, val_logdet, val_dist = validate_model(val_loader, val_dataset, model, beta, config,
                                                                         device)

            # for data in val_loader:
            #     input, label = data
            #
            #     input = val_dataset.format_data(input, device, add_feature=config["add_feature"])
            #     label = label.to(device)
            #
            #     log_p, distloss, logdet, z = model(input, label)
            #
            #     logdet = logdet.mean()
            #
            #     nll_loss, log_p, log_det = val_dataset.format_loss(log_p, logdet)
            #     loss = nll_loss - beta * distloss
            #     val_loss += loss.cpu().numpy()
            #     val_steps += 1
            #     val_nll += nll_loss.cpu().numpy()
            #     val_logp += log_p.cpu().numpy()
            #     val_logdet += log_det.cpu().numpy()
            #     val_dist += distloss.cpu().numpy()
            #
            #     if dataset.is_regression_dataset():
            #         # accuracy
            #         means = model.means.detach().cpu().numpy()
            #         np_z = z.detach().cpu().numpy()
            #         np_label = label.detach().cpu().numpy()
            #         proj, dot_val = project_between(np_z, means[0], means[1])
            #         pred = dot_val.squeeze() * (model.label_max - model.label_min) + model.label_min
            #         accuracy += np.power((pred - np_label), 2).mean()
            #     else:
            #         # accuracy += val_loss
            #         accuracy += val_logp

        model.train()
        # if accuracy > best_accuracy and epoch > args.save_at_epoch:
        if accuracy < best_accuracy and epoch > args.save_at_epoch and epoch % args.save_each_epoch == 0:
            best_accuracy = accuracy
            if is_raytuning:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
            else:
                checkpoint_dir = os.path.join(save_dir, f"checkpoint_{str(epoch).zfill(5)}")
                create_folder(checkpoint_dir)
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(model.state_dict(), path)

            if epoch > 1:
                test_pred_model(args, train_dataset, test_dataset, model, checkpoint_dir, config)

        # if epoch > args.save_at_epoch and epoch % args.save_each_epoch == 0:
        #     with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #         path = os.path.join(checkpoint_dir, "checkpoint")
        #         torch.save((model.state_dict(), optimizer.state_dict()), path)

        if is_raytuning:
            tune.report(loss=(val_loss / val_steps), accuracy=(accuracy / val_steps), nll=(val_nll / val_steps),
                        logp=(val_logp / val_steps), logdet=(val_logdet / val_steps), distloss=(val_dist / val_steps))
        else:
            if args.use_tb:
                loss_dict = {"tot_loss": (val_loss / val_steps),
                             "NLL": (val_nll / val_steps),
                             "logP": (val_logp / val_steps),
                             "logdet": (val_logdet / val_steps),
                             "distloss": (val_dist / val_steps)}
                write_dict_to_tensorboard(writer, loss_dict, base_name="VAL", iteration=epoch)


def test_pred_model(args, train_dataset, test_dataset, model, save_dir, config):
    batch_size = config["batch_size"]

    # reduce train dataset size (fitting too long)
    if args.reduce_test_dataset_size is not None:
        t_dataset = train_dataset.duplicate()
        print('Train dataset reduced in order to accelerate. (stratified)')
        t_dataset.reduce_dataset_ratio(args.reduce_test_dataset_size, stratified=True)
    else:
        t_dataset = train_dataset

    if isinstance(t_dataset, GraphDataset):
        t_dataset.permute_graphs_in_dataset()
        test_dataset.permute_graphs_in_dataset()

    model.eval()
    Z = []
    tlabels = []
    if isinstance(t_dataset, GraphDataset):
        n_permutation = args.n_permutation_test
        for n_perm in range(n_permutation):
            t_dataset.permute_graphs_in_dataset()

            loader = t_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

            with torch.no_grad():
                for j, data in enumerate(loader):
                    inp, labels = data
                    inp = t_dataset.format_data(inp, device, add_feature=config["add_feature"])
                    labels = labels.to(device)
                    log_p, distloss, logdet, out = model(inp, labels)
                    Z.append(out.detach().cpu().numpy())
                    tlabels.append(labels.detach().cpu().numpy())
    else:
        loader = t_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)

        with torch.no_grad():
            for j, data in enumerate(loader):
                inp, labels = data
                inp = t_dataset.format_data(inp, device, add_feature=config["add_feature"])
                labels = labels.to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                Z.append(out.detach().cpu().numpy())
                tlabels.append(labels.detach().cpu().numpy())
    tlabels = np.concatenate(tlabels, axis=0)
    Z = np.concatenate(Z, axis=0).reshape(tlabels.shape[0], -1)

    # Learn SVC
    kernel_name = 'zlinear'
    scaler = False
    if train_dataset.is_regression_dataset():
        param_gridlin = [{'Ridge__alpha': np.concatenate((np.logspace(-5, 2, 11), np.array([1])))}]
        predictor = learn_or_load_modelhyperparams(Z, tlabels, kernel_name, param_gridlin, save_dir,
                                                   model_type=('Ridge', Ridge(max_iter=100000)), scaler=scaler,
                                                   save=False)
    else:
        param_gridlin = [
            {'SVC__kernel': ['linear'], 'SVC__C': np.concatenate((np.logspace(-5, 3, 10), np.array([1])))}]
        model_type = ('SVC', SVC(max_iter=100000))
        # model_type = ('SVC', SVC())
        scaler = False
        predictor = learn_or_load_modelhyperparams(Z, tlabels, kernel_name, param_gridlin, save_dir,
                                                   model_type=model_type, scaler=scaler, save=False)

    print(predictor)

    if isinstance(t_dataset, GraphDataset):
        n_permutation = 20
        our_preds = []
        our_scores = []

        for n_perm in range(n_permutation):
            test_dataset.permute_graphs_in_dataset()
            # OUR APPROACH EVALUATION
            test_loader = test_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)
            test_inZ = []
            elabels = []
            with torch.no_grad():
                for j, data in enumerate(test_loader):
                    inp, labels = data
                    inp = test_dataset.format_data(inp, device, add_feature=config["add_feature"])
                    labels = labels.to(device)
                    log_p, distloss, logdet, out = model(inp, labels)
                    test_inZ.append(out.detach().cpu().numpy())
                    elabels.append(labels.detach().cpu().numpy())
            test_inZ = np.concatenate(test_inZ, axis=0).reshape(len(test_dataset), -1)
            elabels = np.concatenate(elabels, axis=0)

            z_pred = predictor.predict(test_inZ)
            if test_dataset.is_regression_dataset():
                score = np.power(z_pred - elabels, 2).mean()  # MSE
            else:
                score = classification_score(z_pred, elabels)

            our_preds.append(z_pred)
            our_scores.append(score)

        # PRINT RESULTS
        lines = []
        print('Predictions scores :')

        mean_score = np.mean(our_scores)
        # mean_pred_score = np.mean(our_pred_scores)
        std_score = np.std(our_scores)
        score_str = f'Our approach {our_scores} \n' \
                    f'Mean Scores: {mean_score} \n' \
                    f'Std Scores: {std_score}'
        print(score_str)
        lines += [score_str, '\n']

    else:
        test_loader = test_dataset.get_loader(batch_size, shuffle=False, drop_last=False, pin_memory=False)
        test_inZ = []
        elabels = []
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                inp, labels = data
                inp = test_dataset.format_data(inp, device, add_feature=config["add_feature"])
                labels = labels.to(device)
                log_p, distloss, logdet, out = model(inp, labels)
                test_inZ.append(out.detach().cpu().numpy())
                elabels.append(labels.detach().cpu().numpy())
        test_inZ = np.concatenate(test_inZ, axis=0).reshape(len(test_dataset), -1)
        elabels = np.concatenate(elabels, axis=0)

        z_pred = predictor.predict(test_inZ)
        if test_dataset.is_regression_dataset():
            score = np.power(z_pred - elabels, 2).mean()  # MSE
        else:
            score = classification_score(z_pred, elabels)

        lines = []
        print('Predictions scores :')

        score_str = f'Our approach: {score}'
        print(score_str)
        lines += [score_str, '\n']

    with open(f"{save_dir}/eval_res.txt", 'w') as f:
        f.writelines(lines)

    model.train()


def main(args):
    if args.seed is not None:
        set_seed(args.seed)

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

    # TEST RANDGENDATASET
    # from utils.dataset import SimpleDataset, ImDataset
    # dataset = ImDataset(dataset_name=args.dataset, transform=transform)
    # DATASET #
    # dataset = load_dataset(args, args.dataset, args.model, transform=transform, add_feature=args.add_feature)
    dataset = load_dataset(args, args.dataset, args.model, transform=transform)  # do not give the add_feature here !

    redclass_str = ''
    if args.reduce_class_size:
        dataset.reduce_dataset('every_class', how_many=args.reduce_class_size)
        red_str = f'_redclass{args.reduce_class_size}'

    redlabel_str = ''
    if args.unique_label:
        dataset.reduce_dataset('one_class', label=args.unique_label)
        dataset.true_labels[:] = 0
        redlabel_str = f'_label{args.unique_label}'
    elif args.multi_label is not None:
        labels = list(map(int, args.multi_label.strip('[]').split(',')))
        dataset.reduce_dataset('multi_class', label=labels)
        redlabel_str = f'_multilabel'
        for label in labels:
            redlabel_str += '-' + str(label)

    if args.test > 0:
        # TEST with stratified sample
        train_dataset, test_dataset = dataset.split_dataset(args.test, stratified=True, split_type='test')
        train_dataset.ori_X = train_dataset.X
        train_dataset.ori_true_labels = train_dataset.true_labels
        test_dataset.ori_X = test_dataset.X
        test_dataset.ori_true_labels = test_dataset.true_labels
    else:
        train_dataset = dataset
        test_dataset = None

    if args.validation > 0:
        # TEST with stratified sample
        train_dataset, val_dataset = train_dataset.split_dataset(args.validation, stratified=True, split_type='val')
        train_dataset.ori_X = train_dataset.X
        train_dataset.ori_true_labels = train_dataset.true_labels
        val_dataset.ori_X = val_dataset.X
        val_dataset.ori_true_labels = val_dataset.true_labels
    else:
        val_dataset = None

    device = 'cuda:0'

    dim_per_label = dataset.get_dim_per_label(add_feature=args.add_feature)
    # n_dim = dataset.get_n_dim()
    #
    # # initialize gaussian params
    # if not dataset.is_regression_dataset():
    #     if not args.dim_per_label:
    #         uni = np.unique(dataset.true_labels)
    #         dim_per_label = math.floor(n_dim / len(uni))
    #     else:
    #         dim_per_label = args.dim_per_label
    # else:
    #     if not args.dim_per_label:
    #         dim_per_label = n_dim
    #     else:
    #         dim_per_label = args.dim_per_label

    # fixed_eigval = None
    # eigval_str = ''
    if args.set_eigval_manually is None:
        mean_of_eigval_str = str(args.mean_of_eigval).replace('.', '-')
        if args.uniform_eigval:
            var_type = 'uniform'
            #         eigval_list = [args.mean_of_eigval for i in range(dim_per_label)]
            eigval_str = f'_eigvaluniform{mean_of_eigval_str}'
        elif args.gaussian_eigval is not None:
            var_type = 'gaussian'
            g_param = list(map(float, args.gaussian_eigval.strip('[]').split(',')))
            #         assert len(g_param) == 2, 'gaussian_eigval should be composed of 2 float the mean and the std'
            #         import scipy.stats as st
            #
            #         dist = st.norm(loc=g_param[0], scale=g_param[1])
            #         border = 1.6
            #         # step = border * 2 / dim_per_label
            #         x = np.linspace(-border, border, dim_per_label)
            #         # if (dim_per_label % 2) != 0 else np.concatenate(
            #         # (np.linspace(-border, g_param[0], int(dim_per_label / 2))[:-1], [g_param[0]],
            #         #  np.linspace(step, border, int(dim_per_label / 2))))
            #         eigval_list = dist.pdf(x)
            #         mean_eigval = args.mean_of_eigval
            #         a = mean_eigval * dim_per_label / eigval_list.sum()
            #         eigval_list = a * eigval_list
            #         eigval_list[np.where(eigval_list < 1)] = 1
            std_str = str(g_param[1]).replace('.', '-')
            eigval_str = f'_eigvalgaussian{mean_of_eigval_str}std{std_str}'
        else:
            assert False, 'No distribution selected; use uniform_eigval or gaussian_eigval arguments'
    else:
        var_type = 'manual'
        #     fixed_eigval = list(map(float, args.set_eigval_manually.strip('[]').split(',')))
        #     eigval_list = None
        eigval_str = args.set_eigval_manually.strip('[]').replace(',', '-')
        eigval_str = f'_manualeigval{eigval_str}'
    #
    # if not dataset.is_regression_dataset():
    #     gaussian_params = initialize_class_gaussian_params(dataset, eigval_list, isotrope=args.isotrope_gaussian,
    #                                                        dim_per_label=dim_per_label, fixed_eigval=fixed_eigval,
    #                                                        split_graph_dim=args.split_graph_dim)
    # else:
    #     if args.method == 0:
    #         gaussian_params = initialize_regression_gaussian_params(dataset, eigval_list,
    #                                                                 isotrope=args.isotrope_gaussian,
    #                                                                 dim_per_label=dim_per_label,
    #                                                                 fixed_eigval=fixed_eigval)
    #     elif args.method == 1:
    #         gaussian_params = initialize_tmp_regression_gaussian_params(dataset, eigval_list)
    #     elif args.method == 2:
    #         gaussian_params = initialize_tmp_regression_gaussian_params(dataset, eigval_list, ones=True)
    #     else:
    #         assert False, 'no method selected'

    reg_use_var_str = f''
    if args.reg_use_var:
        reg_use_var_str = f'_usevar'

    split_graph_dim_str = f''
    if args.split_graph_dim and isinstance(dataset, GraphDataset) and not dataset.is_regression_dataset():
        split_graph_dim_str = f'_splitgraphdim'

    folder_path = f'{args.dataset}/{args.model}/'
    if args.model == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        #     n_channel = dataset.n_channel
        #     model_single = load_cglow_model(args_cglow, n_channel, gaussian_params=gaussian_params,
        #                                     learn_mean=not args.fix_mean, device=device)
        folder_path += f'b{args_cglow.n_block}_f{args_cglow.n_flow}'
    elif args.model == 'seqflow':
        args_seqflow, _ = seqflow_arguments().parse_known_args()
        #     model_single = load_seqflow_model(dataset.in_size, args_seqflow.n_flow, gaussian_params=gaussian_params,
        #                                       learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
        folder_path += f'f{args_seqflow.n_flow}'
    elif args.model == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        #     # args_ffjord.n_block = args.n_block
        #     model_single = load_ffjord_model(args_ffjord, dataset.in_size, gaussian_params=gaussian_params,
        #                                      learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
        folder_path += f'b{args_ffjord.n_block}'
    elif args.model == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        #     model_single = load_moflow_model(args_moflow, gaussian_params=gaussian_params,
        #                                      learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
        folder_path += f'B_b{args_moflow.b_n_block}_f{args_moflow.b_n_flow}_A_b{args_moflow.a_n_block}_f{args_moflow.a_n_flow}'
        noise_scale = args_moflow.noise_scale
        noise_scale_x = args_moflow.noise_scale_x
    # elif args.model == 'graphnvp':
    #     args_graphnvp, _ = graphnvp_arguments().parse_known_args()
    #     model_single = load_graphnvp_model(args_graphnvp, gaussian_params=gaussian_params,
    #                                       learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
    else:
        assert False, 'unknown model type'

    # model_single.init_means_to_points(train_dataset)

    lmean_str = f'_lmean{args.beta}' if not args.fix_mean else ''
    isotrope_str = '_isotrope' if args.isotrope_gaussian else ''
    add_feature_str = f'_addfeature{args.add_feature}' if args.add_feature is not None else ''
    folder_path += f'_nfkernel{lmean_str}{isotrope_str}{eigval_str}{noise_str}' \
                   f'{redclass_str}{redlabel_str}_dimperlab{dim_per_label}{reg_use_var_str}' \
                   f'{split_graph_dim_str}{add_feature_str}'

    if args.add_in_name_folder is not None:
        folder_path += f'_{args.add_in_name_folder}'

    save_dir = f'./checkpoint/{folder_path}'
    create_folder(save_dir)

    # path = f'./checkpoint/{folder_path}/train_idx'
    # train_dataset.save_split(path)
    # if val_dataset is not None:
    #     path = f'./checkpoint/{folder_path}/val_idx'
    #     val_dataset.save_split(path)

    # restart = args.restart
    # if restart:
    #     from utils.custom_glow import WrappedModel
    #
    #     flow_path = f'./checkpoint/{folder_path}/restart.pth'
    #     model_single = WrappedModel(model_single)
    #     model_single.load_state_dict(torch.load(flow_path, map_location=device))
    #     model_single = model_single.module
    #     folder_path += '/restart'
    #     save_dir = f'./checkpoint/{folder_path}'
    #     create_folder(save_dir)

    config = {
        "var_type": var_type,
        "var": args.mean_of_eigval,
        "beta": args.beta,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "add_feature": args.add_feature,
        "split_graph_dim": args.split_graph_dim
    }
    if args.model == 'moflow':
        config["noise"] = noise_scale
        config["noise_x"] = noise_scale_x

    train_start = time.time()
    # train(args, model_single, folder_path, train_dataset=train_dataset, val_dataset=val_dataset)
    train(args, config, train_dataset, val_dataset, test_dataset, save_dir, is_raytuning=False)
    train_end = time.time()
    # del dataset
    gpu_info = torch.cuda.mem_get_info(device=device)
    # model_single.del_model_from_gpu()
    return f'./checkpoint/{folder_path}', train_end - train_start, gpu_info


if __name__ == "__main__":
    args, _ = training_arguments().parse_known_args()
    print(args)

    main(args)
