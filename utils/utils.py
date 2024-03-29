import random
import torch
import os
import math
import numpy as np
from PIL import Image
import glob
import matplotlib.gridspec as gridspec

from utils.density import multivariate_gaussian
from utils.dataset import GraphDataset

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_dict_to_tensorboard(writer, val_dict, base_name, iteration):
    for name, val in val_dict.items():
        if isinstance(val, dict):
            write_dict_to_tensorboard(writer, val, base_name=base_name + "/" + name, iteration=iteration)
        elif isinstance(val, (list, np.ndarray)):
            continue
        elif isinstance(val, (int, float)):
            writer.add_scalar(base_name + "/" + name, val, iteration)
        else:
            print("Skipping output \"" + str(name) + "\" of value " + str(val) + "(%s)" % (val.__class__.__name__))


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False


def save_every_pic(path, numpy_pics, methods, labels, add_str=None, clamp_min=0, clamp_max=255, rgb=False):
    create_folder(path)
    add_str = '' if add_str is None else f'_{add_str}'
    numpy_pics = np.clip(numpy_pics, clamp_min, clamp_max)
    for i, pic in enumerate(numpy_pics):
        im = pic.squeeze().astype(np.uint8)
        if rgb:
            im = Image.fromarray(im, mode='RGB')
        else:
            im = Image.fromarray(im, mode='L')
        name = f"{methods[i]}_{str(labels[i])}{add_str}.png"
        im.save(f'{path}/{name}')


def save_fig(X, labels, save_path, limits=None, size=7, rangelabels=None, eps=False):
    if rangelabels is not None:
        vmin = rangelabels[0]
        vmax = rangelabels[-1]
        plt.scatter(X[:, 0], X[:, 1], s=size, c=labels, zorder=3, vmin=vmin, vmax=vmax)
    else:
        plt.scatter(X[:, 0], X[:, 1], s=size, c=labels, zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")

    if limits:
        x_xmin, x_xmax, x_ymin, x_ymax = limits
        plt.xlim([x_xmin - 1, x_xmax + 1])
        plt.ylim([x_ymin - 1, x_ymax + 1])

    if eps:
        plt.savefig(fname=f'{save_path}.eps', format='eps')
    plt.savefig(fname=f'{save_path}.png', format='png')

    plt.close()


def save_projection_fig(X, reconstructed_X, labels, label_max, save_path, limits=None, size=7, noisy_X=None):
    # to_show = np.concatenate((X, reconstructed_X), axis=0)
    # c_labels = np.zeros(to_show.shape[0])
    # c_labels[:X.shape[0]] = labels
    # c_labels[X.shape[0]:X.shape[0] + reconstructed_X.shape[0]] = label_max + 1
    # # c_labels[-1 * interp_inX.shape[0]] = label_max + 2
    #
    # plt.scatter(to_show[:, 0], to_show[:, 1], s=7, c=c_labels, zorder=2)
    # plt.xlabel("x")
    # plt.ylabel("y")

    plt.scatter(X[:, 0], X[:, 1], s=size, c=labels, zorder=3)
    plt.xlabel("x")
    plt.ylabel("y")

    c_labels = np.zeros(reconstructed_X.shape[0])
    c_labels[:reconstructed_X.shape[0]] = label_max + 1
    plt.scatter(reconstructed_X[:, 0], reconstructed_X[:, 1], s=size, c='r', zorder=2)
    plt.xlabel("x")
    plt.ylabel("y")

    if noisy_X is not None:
        plt.scatter(noisy_X[:, 0], noisy_X[:, 1], s=size, c=labels, zorder=3)
        plt.xlabel("x")
        plt.ylabel("y")

    # size = 25
    # plt.scatter(noisy_data[:, 0], noisy_data[:, 1], s=size, c='darkorange', zorder=3)
    # plt.scatter(interp_inX[1:, 0], interp_inX[1:, 1], s=size, c='red', zorder=3)
    # plt.scatter(interp_inX[-1, 0], interp_inX[-1, 1], s=size, c='yellow', zorder=3)

    # Links
    to_show = np.zeros((2 * X.shape[0], X.shape[-1]))
    print(to_show.shape)
    for i in range(0, X.shape[0]):
        to_show[2 * i, :] = X[i, :]
        to_show[2 * i + 1, :] = reconstructed_X[i, :]

    for i in range(0, X.shape[0]):
        plt.plot(to_show[2 * i:2 * i + 2, 0], to_show[2 * i:2 * i + 2, 1], color='lime', zorder=1)

    if limits:
        x_xmin, x_xmax, x_ymin, x_ymax = limits
        plt.xlim([x_xmin - 1, x_xmax + 1])
        plt.ylim([x_ymin - 1, x_ymax + 1])

    plt.savefig(fname=f'{save_path}.eps', format='eps')
    plt.savefig(fname=f'{save_path}.png', format='png')
    # Grid
    # plt.scatter(z_grid_inX[:, 0], z_grid_inX[:, 1], c=color, s=5, alpha=0.3)
    plt.close()


def organise_data(path, nrow, method=None, format='png', type='horizontal', last_part=None, order=None):
    images = []
    if last_part is None:
        last_part = path.split('/')[-1]
        last_part = '_' + last_part
    files = sorted(glob.glob(f"{path}/*{last_part}.{format}"))
    if order is not None:
        assert method is not None, 'if order then method should be given'
        files_ordered = []
        for lab in order:
            for file in files:
                if (method in file.split('/')[-1] and method + file.split('/')[-1].split(method)[1].split('_')[
                    0] == lab) or file.split('/')[-1].split('_')[0] == lab:
                    files_ordered.append(file)
        files = files_ordered
    for file in files:
        image = Image.open(file)
        data = np.asarray(image)
        # if len(data.shape) < 3:
        data = np.expand_dims(data, axis=0)
        images.append(data)
    images = np.concatenate(images, axis=0).astype(np.float32)
    images = np.expand_dims(images, axis=1) if len(images.shape) < 4 else images.transpose(0, 3, 1, 2)

    if type == 'vertical':
        ordered = []
        ncol = nrow
        nrow = math.floor(len(files) / ncol)
        for i in range(ncol):
            for j in range(nrow):
                ind = i + j * ncol
                ordered.append(np.expand_dims(images[ind], axis=0))
        ordered = np.concatenate(ordered, axis=0)
        images = ordered

    return images


def format(images, nrow, save_path, row_legends=None, col_legends=None, show=False, res_name=None, dark_mode=False):
    if dark_mode:
        plt.style.use('dark_background')

    if col_legends is not None:
        ncol = len(col_legends)
    else:
        ncol = math.ceil(images.shape[0] / nrow)
    ratios = [0.25] + [1] * nrow if col_legends is not None else [1] * nrow

    col_add = 1 if row_legends is not None else 0
    row_add = 1 if col_legends is not None else 0

    im_size1 = images.shape[-2]
    im_size2 = images.shape[-1]
    fig = plt.figure(figsize=(2 * im_size2 * ncol / 100, 2 * im_size2 * nrow / 100))
    gspec = gridspec.GridSpec(
        ncols=ncol + col_add, nrows=nrow + row_add, figure=fig, height_ratios=ratios, wspace=0, hspace=0
    )
    cmap = plt.get_cmap("gist_gray")

    for row in range(0, nrow):
        if row_legends is not None:
            ax = plt.subplot(
                gspec[row + row_add, 0], frameon=False, xlim=[0, 1], xticks=[], ylim=[0, 1], yticks=[]
            )
            ax.text(
                1,
                0.7,
                row_legends[row],
                family="Roboto Condensed",
                horizontalalignment="right",
                verticalalignment="top",
            )

        for col in range(0, ncol):
            ax = plt.subplot(
                gspec[row + row_add, col + col_add],
                aspect=1,
                frameon=False,
                xlim=[0, 1],
                xticks=[],
                ylim=[0, 1],
                yticks=[],
            )
            if row * ncol + col < images.shape[0]:
                ax.imshow(images[row * ncol + col].transpose(1, 2, 0).astype(np.uint8), cmap=cmap, extent=[0, 1, 0, 1],
                          vmin=0, vmax=255)
            if col_legends is not None and row == 0:
                ax.text(
                    0.5,
                    1.1,
                    col_legends[col],
                    ha="center",
                    va="bottom",
                    size="small",
                    weight="bold",
                )
    create_folder(f'{save_path}')
    name = res_name if res_name is not None else 'grid'
    plt.savefig(f"{save_path}/{name}.png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_class_gaussian_params(dataset, al_list, isotrope=False, dim_per_label=30, fixed_eigval=None,
                                     split_graph_dim=False, add_feature=None):
    uni = np.unique(dataset.true_labels)
    if isinstance(dataset, GraphDataset) and split_graph_dim:
        n_x, n_adj = dataset.calculate_dims(add_feature=add_feature)
        dim_per_label_x = math.floor(n_x / len(uni))
        dim_per_label_adj = math.floor(n_adj / len(uni))
        n_dim = n_x + n_adj
        al_list = al_list[:dim_per_label_x + dim_per_label_adj]
        dim_per_label = dim_per_label_x + dim_per_label_adj
    else:
        n_dim = dataset.get_n_dim(add_feature=add_feature)
    gaussian_params = []
    eigenvecs = np.zeros((n_dim, n_dim))
    np.fill_diagonal(eigenvecs, 1)
    if isotrope:
        for i, label in enumerate(uni):
            mean = np.zeros(n_dim)
            eigenvals = np.ones(n_dim)
            gaussian_params.append((mean, eigenvecs, eigenvals, label))
    else:
        for i, label in enumerate(uni):
            mean = np.zeros(n_dim)
            if fixed_eigval is None:
                # be = np.power(1 / (math.pow(sum(al_list) / len(al_list), dim_per_label)), 1 / (n_dim - dim_per_label))
                # log to calculate
                be = math.exp(1 / (n_dim - dim_per_label) * (- dim_per_label * math.log(sum(al_list) / len(al_list))))
                eigenvals = np.ones(n_dim) * be
                if isinstance(dataset, GraphDataset) and split_graph_dim:
                    eigenvals[dim_per_label_x * i:dim_per_label_x * (i + 1)] = al_list[:dim_per_label_x]
                    eigenvals[n_x + dim_per_label_adj * i:n_x + dim_per_label_adj * (i + 1)] \
                        = al_list[dim_per_label_x:dim_per_label_x + dim_per_label_adj]
                else:
                    eigenvals[dim_per_label * i:dim_per_label * (i + 1)] = al_list
            else:
                eigenvals = np.ones(n_dim)
                eigenvals[:] = fixed_eigval
            gaussian_params.append((mean, eigenvecs, eigenvals, label))

    return gaussian_params


def initialize_regression_gaussian_params(dataset, al_list, isotrope=False, dim_per_label=30, fixed_eigval=None,
                                          add_feature=None):
    n_dim = dataset.get_n_dim(add_feature=add_feature)
    gaussian_params = []
    if isotrope:
        for i in range(2):
            mean = np.zeros(n_dim)
            # mean = np.ones(n_dim) * -20 * i
            eigenvecs = np.zeros((n_dim, n_dim))
            np.fill_diagonal(eigenvecs, 1)
            eigenvals = np.ones(n_dim)
            eigenvals[:] = al_list
            gaussian_params.append((mean, eigenvecs, eigenvals, i))
    else:
        assert n_dim != dim_per_label, 'for regression dataset, isotrope_gaussian should be used if n_dim_per_label ' \
                                       'is not defined and therefore n_dim_per_label = n_dim'
        for i in range(2):
            mean = np.zeros(n_dim)
            eigenvecs = np.zeros((n_dim, n_dim))
            np.fill_diagonal(eigenvecs, 1)
            if fixed_eigval is None:
                # be = np.power(1 / (math.pow(sum(al_list) / len(al_list), dim_per_label)), 1 / (n_dim - dim_per_label))
                # log to calculate
                be = math.exp(1 / (n_dim - dim_per_label) * (- dim_per_label * math.log(sum(al_list) / len(al_list))))
                eigenvals = np.ones(n_dim) * be
                eigenvals[dim_per_label * i:dim_per_label * (i + 1)] = al_list
            else:
                eigenvals = np.ones(n_dim)
                eigenvals[:] = fixed_eigval
            gaussian_params.append((mean, eigenvecs, eigenvals, i))

    return gaussian_params


def initialize_tmp_regression_gaussian_params(dataset, al_list, ones=False):
    n_dim = dataset.X[0].shape[0]
    for sh in dataset.X[0].shape[1:]:
        n_dim *= sh
    gaussian_params = []

    assert len(al_list) == 1, 'al_list should contains one dimension only'
    dim_interpolation = 1

    for i in range(2):
        mean = np.zeros(n_dim)
        eigenvecs = np.zeros((n_dim, n_dim))
        np.fill_diagonal(eigenvecs, 1)
        be = np.exp(
            1 / (n_dim - dim_interpolation) * np.log(1 / math.pow(sum(al_list) / len(al_list), dim_interpolation)))
        eigenvals = np.ones(n_dim) * be if not ones else np.ones(n_dim) * 0.1
        eigenvals[:1] = al_list
        mean[:1] = i
        gaussian_params.append((mean, eigenvecs, eigenvals, i))

    return gaussian_params


def calculate_log_p_with_gaussian_params(x, label, means, gaussian_params):
    log_ps = []
    for i, gaussian_param in enumerate(gaussian_params):
        log_ps.append(multivariate_gaussian(x, mean=means[i], determinant=gaussian_param[1],
                                            inverse_cov_mat_diag=gaussian_param[0]).unsqueeze(1))

    log_ps = torch.cat(log_ps, dim=1)
    one_hot_label = torch.nn.functional.one_hot(label, num_classes=log_ps.shape[1])
    log_p = torch.sum(log_ps * one_hot_label, dim=1)

    # # Change in calculation test
    # np_label = label.clone().detach().cpu().numpy()
    # test = np.array(gaussian_params, dtype=object)[np_label]
    #
    # determinant = torch.from_numpy(test[:, 1].astype(np.float32)).to(x.device)
    # diag_inv = torch.diag_embed(
    #     torch.from_numpy(np.concatenate([np.expand_dims(v, 0) for v in test[:, 0]], 0).astype(np.float32))).to(x.device)
    # mean = means[np_label].to(torch.float32)
    #
    # b_size = x.shape[0]
    # z_flat = x.reshape(b_size, -1)
    # k = z_flat.shape[1]
    # log_p = -0.5 * (k * math.log(2 * math.pi) + torch.diag(
    #     (torch.diagonal((z_flat - mean) @ diag_inv, offset=0, dim1=0, dim2=1).transpose(1, 0)
    #      .unsqueeze(0) @ torch.transpose(z_flat - mean, 1, 0))
    #         .reshape(b_size, -1), 0)) - torch.log(determinant)

    return log_p


def predict_by_log_p_with_gaussian_params(x, means, gaussian_params):
    # log_ps = []
    # for i, gaussian_param in enumerate(gaussian_params):
    #     log_ps.append(multivariate_gaussian(x, mean=means[i], determinant=gaussian_param[1],
    #                                         inverse_cov_mat_diag=gaussian_param[0]).unsqueeze(1))
    #
    # log_ps = torch.cat(log_ps, dim=1)
    # one_hot_label = torch.nn.functional.one_hot(label, num_classes=log_ps.shape[1])
    # log_p = torch.sum(log_ps * one_hot_label, dim=1)

    # Change in calculation test
    # for each label
    log_ps = []
    for label in range(means.shape[0]):
        # np_label = label.clone().detach().cpu().numpy()
        test = np.array(gaussian_params, dtype=object)[label]

        determinant = torch.from_numpy(np.repeat(np.expand_dims(test[1].astype(np.float32), 0), x.shape[0])).to(
            x.device)
        diag_inv = torch.diag_embed(test[0].unsqueeze(0).repeat(x.shape[0], 1)).to(torch.float32).to(x.device)
        mean = means[label].to(torch.float32)

        b_size = x.shape[0]
        z_flat = x.reshape(b_size, -1)
        k = z_flat.shape[1]
        log_p = -0.5 * (k * math.log(2 * math.pi) + torch.diag(
            (torch.diagonal((z_flat - mean) @ diag_inv, offset=0, dim1=0, dim2=1).transpose(1, 0)
             .unsqueeze(0) @ torch.transpose(z_flat - mean, 1, 0))
                .reshape(b_size, -1), 0)) - torch.log(determinant)

        log_ps.append(log_p.unsqueeze(0))
    result = torch.argmax(torch.cat(log_ps, 0), 0)
    return result


# def calculate_log_p_with_gaussian_params_regression(x, label, means, gaussian_params, label_min, label_max):
#     log_ps = []
#     for i, gaussian_param in enumerate(gaussian_params):
#         log_ps.append(multivariate_gaussian(x, mean=means[i], determinant=gaussian_param[1],
#                                             inverse_covariance_matrix=gaussian_param[0]).unsqueeze(1))
#
#     lab_fac = ((label - label_min) / (label_max - label_min)).unsqueeze(1)
#     log_p = log_ps[0] * lab_fac + log_ps[1] * (1 - lab_fac)
#
#     return log_p


def calculate_log_p_with_gaussian_params_regression(x, mean, inv_cov, det):
    log_ps = multivariate_gaussian(x, mean=mean, determinant=det,
                                   inverse_cov_mat_diag=inv_cov).unsqueeze(1)

    return log_ps


def load_dataset(args, dataset_name, is_graph_model, to_evaluate=False, transform=None, add_feature=None):
    from utils.dataset import ImDataset, SimpleDataset, RegressionGraphDataset, ClassificationGraphDataset, \
        SIMPLE_DATASETS, SIMPLE_REGRESSION_DATASETS, IMAGE_DATASETS, GRAPH_REGRESSION_DATASETS, \
        GRAPH_CLASSIFICATION_DATASETS

    if transform is None:
        if to_evaluate:
            if dataset_name == 'cifar10':
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            if dataset_name in GRAPH_REGRESSION_DATASETS or dataset_name in GRAPH_CLASSIFICATION_DATASETS or (
                    dataset_name == 'fishtoxi' and is_graph_model):
                transform = 'permutation'

    # DATASET #
    if dataset_name in IMAGE_DATASETS:
        dataset = ImDataset(dataset_name=dataset_name, n_bits=args.n_bits, transform=transform)
    elif dataset_name == 'fishtoxi':  # Special case where the data can be either graph or vectorial data
        if is_graph_model:
            dataset = RegressionGraphDataset(dataset_name=dataset_name, transform=transform, add_feature=add_feature)
        else:
            dataset = SimpleDataset(dataset_name=dataset_name, transform=transform, add_feature=add_feature)
    elif dataset_name in SIMPLE_DATASETS or dataset_name in SIMPLE_REGRESSION_DATASETS:
        dataset = SimpleDataset(dataset_name=dataset_name, transform=transform, add_feature=add_feature)
    elif dataset_name in GRAPH_REGRESSION_DATASETS:
        dataset = RegressionGraphDataset(dataset_name=dataset_name, transform=transform, add_feature=add_feature)
    elif dataset_name in GRAPH_CLASSIFICATION_DATASETS:
        dataset = ClassificationGraphDataset(dataset_name=dataset_name, transform=transform, add_feature=add_feature)
    else:
        assert False, 'unknown dataset'
    return dataset


def visualize_points(data, data_labels, path, limits=None):
    plt.rcParams["font.size"] = 10.0
    plt.rcParams["font.serif"] = ["Source Serif Pro"]
    plt.rcParams["font.sans-serif"] = ["Source Sans Pro"]
    plt.rcParams["font.monospace"] = ["Source Code Pro"]
    plt.rcParams["savefig.format"] = 'png'

    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()

    ax.set_xlabel("X", weight="medium")
    ax.set_ylabel("Y", weight="medium")

    n = data.shape[0]
    np.random.seed(1)
    X = data[:, 0]
    Y = data[:, 1]
    S = np.ones(n) * 40

    cmap = plt.get_cmap("RdYlBu")
    scatter = plt.scatter(X, Y, s=S, edgecolor="black", linewidth=0.75, zorder=-20)
    scatter = plt.scatter(X, Y, s=S, edgecolor="None", facecolor="white", zorder=-10)
    scatter = plt.scatter(X, Y, c=data_labels, cmap=cmap, edgecolor="None", alpha=0.5)

    if limits is not None:
        plt.xlim([limits[0][0], limits[0][1]])
        plt.ylim([limits[1][0], limits[1][1]])
    plt.savefig(path)
    plt.show()
    plt.close()


def create_animation(dir, how_much_repeat=0, figsize=None):
    # GIF
    import matplotlib.animation as animation

    files = sorted(glob.glob(f"{dir}/*.png"))
    image_array = []

    for my_file in files:
        image = Image.open(my_file)
        image_array.append(image)

    for i in range(how_much_repeat):
        image_array.append(image_array[-1])

    # Create the figure and axes objects
    if figsize is not None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()
    ax.axis('off')

    # Set the initial image
    im = ax.imshow(image_array[0], animated=True)

    def update(i):
        im.set_array(image_array[i])
        return im,

    # Create the animation object
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=100, blit=True,
                                            repeat_delay=400, )

    # Show the animation
    plt.show()
    animation_fig.save(f"{dir}/animation.gif")
    plt.close()
