from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import numpy as np
from numpy import genfromtxt
import math
import copy
import scipy
import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
import scipy
import pickle

from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, make_moons, make_swiss_roll
from utils import visualize_flow

DATASETS = ['single_moon', 'double_moon', 'iris', 'bcancer', 'mnist']
REGRESSION_DATASETS = ['swissroll', 'diabetes', 'waterquality', 'aquatoxi', 'fishtoxi', 'trafficflow']
GRAPH_DATASETS = ['qm7', 'qm9', 'freesolv', 'esol', 'lipo']


# abstract base kernel dataset class
class BaseDataset(Dataset):
    def __init__(self, X, true_labels, test_dataset=None):
        self.ori_X = X
        self.ori_true_labels = true_labels
        self.X = X
        self.true_labels = true_labels

        self.idx = None

        self.reduce_type = 'all'

        self.test_dataset = test_dataset

    def __len__(self):
        return len(self.X)

    def get_flattened_X(self):
        return self.X.reshape(self.X.shape[0], -1)

    def get_n_dim(self):
        n_dim = self.X[0].shape[0]
        for sh in self.X[0].shape[1:]:
            n_dim *= sh
        return n_dim

    def get_loader(self, batch_size, shuffle=True, drop_last=True, pin_memory=True, sampler=False):
        if sampler:
            class_sample_count = np.histogram(self.true_labels)[0]
            idxs = np.argsort(self.true_labels)
            probs = np.zeros(len(self))
            done = 0
            for i in range(len(class_sample_count)):
                probs[idxs[done:done + class_sample_count[i]]] = class_sample_count[i]
                done += class_sample_count[i]
            probs = 1 / torch.Tensor(probs / self.true_labels.shape[0])
            sampler = torch.utils.data.sampler.WeightedRandomSampler(probs, len(self), replacement=True)
            loader = torch.utils.data.DataLoader(self, batch_size=batch_size,
                                                 drop_last=drop_last, pin_memory=pin_memory, sampler=sampler)
        else:
            loader = torch.utils.data.DataLoader(
                dataset=self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=pin_memory)
        # print("Loader using batch size {}. Total {} iterations/epoch.".format(batch_size, len(loader)))
        return loader

    def reduce_dataset(self, reduce_type, label=None, how_many=None, reduce_from_ori=True):
        if reduce_from_ori:
            X = self.ori_X
            true_labels = self.ori_true_labels
        else:
            X = self.X
            true_labels = self.true_labels
        n_X = None
        n_true_labels = None
        self.reduce_type = reduce_type
        n_idx = None
        if reduce_type == 'every_class':
            for c in set(true_labels):
                tmp_X = X[true_labels == c]
                idx = np.where(true_labels == c)[0]
                if how_many is not None:
                    idx = idx[random.sample(range(0, idx.shape[0]), k=how_many)]
                    tmp_X = X[idx]
                    tmp_labels = true_labels[idx]
                else:
                    tmp_labels = true_labels[true_labels == c]

                if n_X is None:
                    n_X = tmp_X
                    n_true_labels = tmp_labels
                    n_idx = idx
                else:
                    n_X = np.concatenate((n_X, tmp_X), axis=0)
                    n_true_labels = np.concatenate((n_true_labels, tmp_labels), axis=0)
                    n_idx = np.concatenate((n_idx, idx), axis=0)
        elif reduce_type == 'one_class':
            assert label is not None, 'to use one_class reduce method, selected_label is needed'
            n_idx = np.where(true_labels == label)[0]
            n_X = X[true_labels == label]
            n_true_labels = true_labels[true_labels == label]
        elif reduce_type == 'multi_class':
            assert label is not None and isinstance(label,
                                                    list), 'to use multi_class reduce method, label should be a list of label'
            n_idx = None
            n_X = None
            n_true_labels = None
            for lab in label:
                idx = np.where(true_labels == lab)[0]
                tmp_X = X[true_labels == lab]
                tmp_labels = true_labels[true_labels == lab]
                if n_X is None:
                    n_idx = idx
                    n_X = tmp_X
                    n_true_labels = tmp_labels
                else:
                    n_idx = np.concatenate((n_idx, idx), axis=0)
                    n_X = np.concatenate((n_X, tmp_X), axis=0)
                    n_true_labels = np.concatenate((n_true_labels, tmp_labels), axis=0)
        else:
            assert False, 'unknown reducing method'

        if not reduce_from_ori and self.idx is not None:
            self.idx = self.idx[n_idx]
        else:
            self.idx = n_idx
        self.X = n_X
        self.true_labels = n_true_labels

    def split_dataset(self, validation, stratified=False):
        val_dataset = copy.deepcopy(self)
        train_dataset = copy.deepcopy(self)
        if self.test_dataset is not None:
            print('Test dataset known, split using the test dataset...')
            val_dataset.X = self.test_dataset[0]
            val_dataset.true_labels = self.test_dataset[1]
            val_dataset.test_dataset = None
            val_dataset.idx = None

            train_dataset.test_dataset = None
        else:
            if stratified:
                class_sample_count = np.histogram(self.true_labels)[0]
                idxs = np.argsort(self.true_labels)
                # probs = np.zeros(len(self))
                # done = 0
                # for i in range(len(class_sample_count)):
                #     probs[idxs[done:done + class_sample_count[i]]] = class_sample_count[i]
                #     done += class_sample_count[i]
                # probs = 1 / torch.Tensor(probs / self.true_labels.shape[0])
                # sampler = torch.utils.data.sampler.WeightedRandomSampler(probs,
                #                                                          math.floor(self.X.shape[0] * validation),
                #                                                          replacement=False)
                # val_idx = []
                # for idx in sampler:
                #     val_idx.append(idx)

                done = 0
                val_idx = []
                for i in range(len(class_sample_count)):
                    nb_idx = math.floor(class_sample_count[i] * validation)
                    if nb_idx == 0 and class_sample_count[i] > 1:
                        nb_idx = 1
                    val_idx += random.sample(list(idxs[done:done + class_sample_count[i]]), k=nb_idx)
                    done += class_sample_count[i]

            else:
                val_idx = random.sample(range(0, self.X.shape[0]), k=math.floor(self.X.shape[0] * validation))

            if isinstance(self.X, list):  # Graphs
                train_idx = [idx for idx in range(len(self.X)) if idx not in val_idx]
                x, adj = list(zip(*self.X))
                x = np.array(x)
                adj = np.array(adj)
                val_x = x[val_idx]
                val_adj = adj[val_idx]
                val_X = []
                for i in range(len(val_idx)):
                    val_X.append((val_x[i], val_adj[i]))
                val_dataset.X = val_X
                train_x = x[train_idx]
                train_adj = adj[train_idx]
                train_X = []
                for i in range(len(train_idx)):
                    train_X.append((train_x[i], train_adj[i]))
                train_dataset.X = train_X

            else:
                val_dataset.X = self.X[val_idx]
                train_idx = [idx for idx in range(self.X.shape[0]) if idx not in val_idx]
                train_dataset.X = self.X[train_idx]
            val_dataset.true_labels = self.true_labels[val_idx]

            train_dataset.true_labels = self.true_labels[train_idx]

            if self.idx is None:
                val_dataset.idx = val_idx
                train_dataset.idx = train_idx
            else:
                val_dataset.idx = self.idx[val_idx]
                train_dataset.idx = self.idx[train_idx]

        return train_dataset, val_dataset

    def reduce_regression_dataset(self, ratio, stratified=True):
        if stratified:
            class_sample_count = np.histogram(self.true_labels)[0]
            idxs = np.argsort(self.true_labels)

            done = 0
            n_idx = []
            for i in range(len(class_sample_count)):
                nb_idx = math.floor(class_sample_count[i] * ratio)
                if nb_idx == 0 and class_sample_count[i] > 1:
                    nb_idx = 1
                n_idx += random.sample(list(idxs[done:done + class_sample_count[i]]), k=nb_idx)
                done += class_sample_count[i]
        else:
            n_idx = random.sample(range(0, self.X.shape[0]), k=math.floor(self.X.shape[0] * ratio))

        if isinstance(self.X, list):  # Graphs
            x, adj = list(zip(*self.X))
            x = np.array(x)
            adj = np.array(adj)
            n_x = x[n_idx]
            n_adj = adj[n_idx]
            n_X = []
            for i in range(len(n_idx)):
                n_X.append((n_x[i], n_adj[i]))
            self.X = n_X
        else:
            self.X = self.X[n_idx]
        self.true_labels = self.true_labels[n_idx]

        if self.idx is None:
            self.idx = n_idx
        else:
            self.idx = self.idx[n_idx]

    def duplicate(self):
        return copy.deepcopy(self)

    def save_split(self, save_path):
        if self.idx is not None:
            np.save(save_path, self.idx)
        else:
            print('The dataset doesn\'t have idx to save!')

    def load_split(self, load_path):
        if os.path.exists(load_path):
            self.idx = np.load(load_path)
            self.X = self.ori_X[self.idx]
            self.true_labels = self.ori_true_labels[self.idx]
        else:
            assert False, f'No file to load split at the path : {load_path}'

    # To implement
    def __getitem__(self, idx):
        raise NotImplementedError


class SimpleDataset(BaseDataset):
    def __init__(self, dataset_name, transform=None):
        self.dataset_name = dataset_name
        self.label_type = np.int if self.dataset_name not in REGRESSION_DATASETS else np.float
        self.transform = transform
        ori_X, ori_true_labels, test_dataset = self.load_dataset(dataset_name)
        super().__init__(ori_X, ori_true_labels, test_dataset)
        print('Z and K are not initialized in constructor')
        self.im_size = ori_X.shape[-1]

        self.current_mode = 'XY'
        self.get_func = self.get_XY_item

        # if self.is_regression_dataset():
        #     self.label_mindist = self.init_labelmin()

    def __getitem__(self, idx):
        return self.get_func(idx)

    # def init_labelmin(self):
    #     # mat = scipy.spatial.distance.cdist(np.expand_dims(self.true_labels, axis=1),
    #     #                                    np.expand_dims(self.true_labels, axis=1))
    #     # mat[np.where(mat == 0)] = 1
    #     # label_mindist = mat.min()
    #
    #     a = np.sort(self.true_labels)
    #     max_dist_neig = 0
    #     sum_dist_neig = 0
    #     for i in range(a.shape[0] - 1):
    #         sum_dist_neig += a[i + 1] - a[i]
    #         if a[i + 1] - a[i] > max_dist_neig:
    #             max_dist_neig = a[i + 1] - a[i]
    #     mean_dist_neig = sum_dist_neig / (a.shape[0] - 1)
    #
    #     return mean_dist_neig
    #     # return label_mindist

    def get_XY_item(self, idx):
        input = self.X[idx]
        if self.transform is not None:
            input = torch.from_numpy(input)
            input = self.transform(input)
        return input, (self.true_labels[idx]).astype(self.label_type)

    def reduce_dataset(self, reduce_type, label=None, how_many=None, reduce_from_ori=True):
        assert self.dataset_name not in REGRESSION_DATASETS, "the dataset can\'t be reduced (regression dataset)"
        return self.reduce_dataset(reduce_type, label, how_many, reduce_from_ori)

    @staticmethod
    def load_dataset(name):
        path = './datasets'
        exist_dataset = os.path.exists(f'{path}/{name}_data.npy') if path is not None else False
        test_dataset = None
        fulldata = True
        name_c = name
        if fulldata and name in ['aquatoxi', 'fishtoxi']:
            name_c = f'{name}_fulldata'
            exist_dataset = os.path.exists(f'{path}/{name_c}_data.npy')
        if exist_dataset:
            X = np.load(f'{path}/{name_c}_data.npy')
            labels = np.load(f'{path}/{name_c}_labels.npy')
            if name == 'single_moon':
                visualize_flow.LOW = -1.5
                visualize_flow.HIGH = 2.5
            elif name == 'double_moon':
                visualize_flow.LOW = -1.5
                visualize_flow.HIGH = 2.5
            elif name == 'iris':
                visualize_flow.LOW = -2.7
                visualize_flow.HIGH = 3.2
            elif name == 'bcancer':
                visualize_flow.LOW = -3.5
                visualize_flow.HIGH = 12.5
            elif name == 'swissroll':
                visualize_flow.LOW = -2
                visualize_flow.HIGH = 2.2
            elif name == 'diabetes':
                visualize_flow.LOW = -3
                visualize_flow.HIGH = 4.3
            elif name == 'waterquality':
                X_test = np.load(f'{path}/{name}_testdata.npy')
                labels_test = np.load(f'{path}/{name}_testlabels.npy')
                test_dataset = (X_test, labels_test)
                visualize_flow.LOW = -3
                visualize_flow.HIGH = 4.3
            elif name == 'aquatoxi':
                if fulldata:
                    X_test = np.load(f'{path}/{name_c}_testdata.npy')
                    labels_test = np.load(f'{path}/{name_c}_testlabels.npy')
                    test_dataset = (X_test, labels_test)
                visualize_flow.LOW = -6
                visualize_flow.HIGH = 14
            elif name == 'fishtoxi':
                if fulldata:
                    X_test = np.load(f'{path}/{name_c}_testdata.npy')
                    labels_test = np.load(f'{path}/{name_c}_testlabels.npy')
                    test_dataset = (X_test, labels_test)
                visualize_flow.LOW = -6
                visualize_flow.HIGH = 14
            elif name == 'trafficflow':
                X_test = np.load(f'{path}/{name}_testdata.npy')
                labels_test = np.load(f'{path}/{name}_testlabels.npy')
                test_dataset = (X_test, labels_test)
                visualize_flow.LOW = -4
                visualize_flow.HIGH = 5.8
        else:
            if name == 'single_moon':
                n_samples = 1000
                X, labels = make_moons(n_samples, shuffle=False, noise=0)  # noise std 0.1
                X = X[np.where(labels == 0)]
                labels = labels[np.where(labels == 0)]
                visualize_flow.LOW = -1.5
                visualize_flow.HIGH = 2.5
            elif name == 'double_moon':
                n_samples = 1000
                X, labels = make_moons(n_samples, shuffle=False, noise=0)  # noise std 0.1
                visualize_flow.LOW = -1.5
                visualize_flow.HIGH = 2.5
            elif name == 'iris':
                X, labels = load_iris(return_X_y=True)  # noise std 0.1
                std = np.std(X, axis=0)
                mean = X.mean(axis=0)
                X = ((X - mean) / std)  # noise std 0.2
                visualize_flow.LOW = -2.7
                visualize_flow.HIGH = 3.2
            elif name == 'bcancer':
                X, labels = load_breast_cancer(return_X_y=True)
                std = np.std(X, axis=0)
                mean = X.mean(axis=0)
                X = ((X - mean) / std)  # noise std 0.2
                visualize_flow.LOW = -3.5
                visualize_flow.HIGH = 12.5
            # REGRESSION DATASETS
            elif name == 'swissroll':
                n_samples = 1000
                X, labels = make_swiss_roll(n_samples=n_samples, noise=0)
                X = X[:, [0, 2]]
                std = np.std(X, axis=0)
                mean = X.mean(axis=0)
                X = ((X - mean) / std)
                visualize_flow.LOW = -2
                visualize_flow.HIGH = 2.2
            elif name == 'diabetes':
                X, labels = load_diabetes(return_X_y=True)
                std = np.std(X, axis=0)
                mean = X.mean(axis=0)
                X = ((X - mean) / std)
                visualize_flow.LOW = -3
                visualize_flow.HIGH = 4.3
            elif name == 'airquality':
                import csv
                results = []
                with open(f"{path}/input.csv") as csvfile:
                    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
                    for row in reader:  # each row is a list
                        results.append(row)
                print(results)
                X = np.loadtxt(f"{path}/AirQualityUCI.csv")
            elif name == 'waterquality':
                import scipy.io

                mat = scipy.io.loadmat(f"{path}/water_quality_prediciton/water_dataset.mat")
                X = np.concatenate(mat['X_tr'].squeeze(), axis=0)
                labels = np.concatenate(mat['Y_tr'].squeeze(), axis=0)
                X_test = np.concatenate(mat['X_te'].squeeze(), axis=0)
                y_test = np.concatenate(mat['Y_te'].squeeze(), axis=0)
                std = np.std(X, axis=0)
                mean = X.mean(axis=0)
                X = ((X - mean) / std)
                X_test = ((X_test - mean) / std)
                visualize_flow.LOW = -9.5
                visualize_flow.HIGH = 9

                test_dataset = (X_test, y_test)
                np.save(f'{path}/{name}_testdata.npy', X_test)
                np.save(f'{path}/{name}_testlabels.npy', y_test)
            elif name == 'aquatoxi':
                if fulldata:
                    name += '_fulldata'
                    import pandas as pd
                    mat = pd.read_excel(f'{path}/qsar_aquatic_toxicity_fulldata.xlsx', engine='openpyxl')
                    mat = np.array(mat)

                    X = mat[np.where(mat[:, 2] == 'train'), 4:].squeeze().astype(np.float64)
                    labels = mat[np.where(mat[:, 2] == 'train'), 3].squeeze().astype(np.float64)
                    X_test = mat[np.where(mat[:, 2] == 'test'), 4:].squeeze().astype(np.float64)
                    y_test = mat[np.where(mat[:, 2] == 'test'), 3].squeeze().astype(np.float64)
                    std = np.std(X, axis=0)
                    mean = X.mean(axis=0)
                    X = ((X - mean) / std)
                    X_test = ((X_test - mean) / std)

                    test_dataset = (X_test, y_test)
                    np.save(f'{path}/{name}_testdata.npy', X_test)
                    np.save(f'{path}/{name}_testlabels.npy', y_test)
                else:
                    mat = np.loadtxt(open(f"{path}/qsar_aquatic_toxicity.csv", "rb"), delimiter=";")

                    X = mat[:, :-1]
                    labels = mat[:, -1]
                    std = np.std(X, axis=0)
                    mean = X.mean(axis=0)
                    X = ((X - mean) / std)
                visualize_flow.LOW = -6
                visualize_flow.HIGH = 14
            elif name == 'fishtoxi':
                if fulldata:
                    name += '_fulldata'
                    import pandas as pd
                    mat = pd.read_excel(f'{path}/qsar_fish_toxicity_fulldata.xlsx', engine='openpyxl')
                    mat = np.array(mat)

                    X = mat[np.where(mat[:, 2] == 'train'), 4:].squeeze().astype(np.float64)
                    labels = mat[np.where(mat[:, 2] == 'train'), 3].squeeze().astype(np.float64)
                    X_test = mat[np.where(mat[:, 2] == 'test'), 4:].squeeze().astype(np.float64)
                    y_test = mat[np.where(mat[:, 2] == 'test'), 3].squeeze().astype(np.float64)
                    std = np.std(X, axis=0)
                    mean = X.mean(axis=0)
                    X = ((X - mean) / std)
                    X_test = ((X_test - mean) / std)

                    test_dataset = (X_test, y_test)
                    np.save(f'{path}/{name}_testdata.npy', X_test)
                    np.save(f'{path}/{name}_testlabels.npy', y_test)
                else:
                    mat = np.loadtxt(open(f"{path}/qsar_fish_toxicity.csv", "rb"), delimiter=";")

                    X = mat[:, :-1]
                    labels = mat[:, -1]
                    std = np.std(X, axis=0)
                    mean = X.mean(axis=0)
                    X = ((X - mean) / std)
                visualize_flow.LOW = -6
                visualize_flow.HIGH = 14
            elif name == 'trafficflow':
                import scipy.io

                mat = scipy.io.loadmat(f"{path}/Traffic Flow Prediction/traffic_dataset.mat")
                X = np.concatenate([np.expand_dims(arr.toarray(), axis=0) for arr in mat['tra_X_tr'].squeeze()],
                                   axis=0).swapaxes(0, 1)
                X = X.reshape(-1, X.shape[-1])
                labels = mat['tra_Y_tr'].reshape(-1, 1).squeeze()
                X_test = np.concatenate([np.expand_dims(arr.toarray(), axis=0) for arr in mat['tra_X_te'].squeeze()],
                                        axis=0).swapaxes(0, 1)
                X_test = X_test.reshape(-1, X_test.shape[-1])
                y_test = mat['tra_Y_te'].reshape(-1, 1).squeeze()
                std = np.std(X, axis=0)
                mean = X.mean(axis=0)
                X = ((X - mean) / std)
                X_test = ((X_test - mean) / std)
                visualize_flow.LOW = -4
                visualize_flow.HIGH = 5.8

                test_dataset = (X_test, y_test)
                np.save(f'{path}/{name}_testdata.npy', X_test)
                np.save(f'{path}/{name}_testlabels.npy', y_test)
            else:
                assert False, 'unknown dataset'

            np.save(f'{path}/{name}_data.npy', X)
            np.save(f'{path}/{name}_labels.npy', labels)

        return X, labels, test_dataset

    @staticmethod
    def format_data(input, n_bits, n_bins, device):
        return input.float().to(device)

    def format_loss(self, log_p, logdet, n_bins):
        loss = logdet + log_p
        return -loss.mean(), log_p.mean(), logdet.mean()

    @staticmethod
    def rescale(x):
        return x

    def is_regression_dataset(self):
        return self.dataset_name in REGRESSION_DATASETS


class ImDataset(BaseDataset):
    def __init__(self, dataset_name, transform=None, test=False, noise_transform=None):
        self.dataset_name = dataset_name
        dset = ImDataset.load_image_dataset(dataset_name, transform, test)
        self.transform = dset.transform
        if self.transform is not None:
            for tr in self.transform.transforms:
                if isinstance(tr, transforms.Normalize):
                    self.norm_mean = np.array(tr.mean).reshape(1, -1, 1, 1)
                    self.norm_std = np.array(tr.std).reshape(1, -1, 1, 1)
                    self.rescale = self.rescale_val_to_im_with_norm
        if not hasattr(self, 'norm_mean'):
            self.rescale = self.rescale_val_to_im_without_norm
        # test with denoise loss
        self.noise_transform = noise_transform
        self.n_channel = dset.data.shape[1]
        self.im_size = dset.data.shape[-1]

        ori_X = dset.data if isinstance(dset.data, np.ndarray) else dset.data.numpy()
        ori_true_labels = dset.targets if isinstance(dset.targets, np.ndarray) else np.array(dset.targets)
        super().__init__(ori_X, ori_true_labels)
        print('Z and K are not initialized in constructor')

        self.current_mode = 'XY'
        self.get_func = self.get_XY_item

        self.get_PIL_image = self.get_PIL_image_L if self.n_channel == 1 else self.get_PIL_image_RGB

    def __getitem__(self, idx):
        return self.get_func(idx)

    def get_PIL_image_RGB(self, idx):
        im = self.X[idx].transpose(1, 2, 0)
        im = Image.fromarray(im.squeeze(), mode='RGB')
        return im

    def get_PIL_image_L(self, idx):
        im = self.X[idx]
        im = Image.fromarray(im.squeeze(), mode='L')
        return im

    def get_XY_item(self, idx):
        target = int(self.true_labels[idx])

        img = self.get_PIL_image(idx)
        if self.transform is not None:
            img = self.transform(img)

        # test with denoise loss
        if self.noise_transform:
            img2 = self.get_PIL_image(idx)
            img2 = self.noise_transform(img2)
            return img, img2, target
        else:
            return img, target

    @staticmethod
    def format_data(input, n_bits, n_bins, device):
        input = input * 255

        if n_bits < 8:
            input = torch.floor(input / 2 ** (8 - n_bits))

        input = input / n_bins - 0.5
        input = input + torch.rand_like(input) / n_bins
        input = input.to(device)
        return input

    def format_loss(self, log_p, logdet, n_bins):
        n_pixel = self.im_size * self.im_size * 3

        loss = -math.log(n_bins) * n_pixel
        loss = loss + logdet + log_p

        return (
            (-loss / (math.log(2) * n_pixel)).mean(),
            (log_p / (math.log(2) * n_pixel)).mean(),
            (logdet / (math.log(2) * n_pixel)).mean(),
        )

    @staticmethod
    def rescale_val_to_im_without_norm(x):
        return x * 255

    def rescale_val_to_im_with_norm(self, x):
        x = (x * self.norm_std + self.norm_mean) * 255
        return x

    @staticmethod
    def load_image_dataset(name, transform=None, test=False):
        if name == 'mnist':
            if transform is None:
                transform = transforms.Compose(
                    [
                        transforms.Resize(28),
                        transforms.CenterCrop(28),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]
                )
            dset = datasets.MNIST(root='./datasets', train=not test, download=True, transform=transform)
            dset.data = dset.data.unsqueeze(1)
        elif name == 'cifar10':
            if transform is None:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            # class CIFAR10:
            #     def __init__(self, train, transform=None):
            #         path = './datasets/cifar-10-python/cifar-10-batches-py/data_batch_1' if train \
            #             else './datasets/cifar-10-python/cifar-10-batches-py/test_batch'
            #         self.train = train
            #         self.dset = CIFAR10.unpickle(path)
            #         keys = list(self.dset.keys())
            #         self.data = self.dset[keys[2]].reshape(-1, 3, 32, 32)
            #         self.targets = np.array(self.dset[keys[1]])
            #         self.transform = transform
            #
            #     @staticmethod
            #     def unpickle(file):
            #         with open(file, 'rb') as fo:
            #             dict = pickle.load(fo, encoding='bytes')
            #         return dict
            #
            # dset = CIFAR10(not test, transform=transform)

            dset = datasets.CIFAR10('./datasets', train=not test, download=True, transform=transform)
            dset.data = dset.data.transpose(0, 3, 1, 2)
        else:
            assert False, 'unknown dataset'

        return dset

    def is_regression_dataset(self):
        return self.dataset_name in REGRESSION_DATASETS


import networkx as nx
from utils.graphs.files_manager import DataLoader
from utils.graphs.molecular_graph_utils import get_molecular_dataset_mp
from utils.graphs.utils_datasets import transform_graph_permutation
import multiprocessing
import itertools

ATOMIC_NUM_MAP = {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}


def chunks(lst, n):
    for i in range(0, len(lst), int(len(lst) / n)):
        yield lst[i:i + int(len(lst) / n)]


def mp_create_graph_func(input_args):
    # input_args is a tuple of ([(x,adj),(x2,adj2),...],label_names dict, create_graph_function)
    chunk, label_names, create_graph_fun = input_args
    return [create_graph_fun(*chunk[i], label_names=label_names, i=i) for i in range(len(chunk))]


class GraphDataset(BaseDataset):
    def __init__(self, dataset_name, transform=None):
        self.dataset_name = dataset_name
        # self.label_type = np.int if self.dataset_name not in REGRESSION_DATASETS else np.float

        (train_dataset, test_dataset), self.label_map = self.load_dataset(dataset_name)
        if len(train_dataset) == 3:
            xs, adjs, y = train_dataset
            xs_test, adjs_test, y_test = test_dataset
        else:
            xs, adjs, self.smiles, y = train_dataset
            xs_test, adjs_test, self.smiles_test, y_test = test_dataset
        y = np.array(y).squeeze()
        test_dataset = (list(zip(xs_test, adjs_test)), np.array(y_test).squeeze())

        self.transform = transform_graph_permutation if transform == 'permutation' else None
        self.atomic_num_list = [ATOMIC_NUM_MAP[label] for label, _ in self.label_map.items()] + [0]
        graphs = list(zip(xs, adjs))

        # Graphs for tests
        self.label_names = {'node_labels': ['atom_type'], 'node_attrs': ['atom_type'], 'edge_labels': ['bond_type'],
                            'edge_attrs': ['bond_type']}
        self.node_labels = self.label_names['node_labels']
        self.node_attrs = self.label_names['node_attrs']
        self.edge_labels = self.label_names['edge_labels']
        self.edge_attrs = self.label_names['edge_attrs']

        super().__init__(graphs, y, test_dataset)
        # self.Gn = self.init_graphs_mp()

        print('Z and K are not initialized in constructor')
        self.im_size = -1

        # if self.is_regression_dataset():
        #     self.label_mindist = self.init_labelmin()

    def get_flattened_X(self):
        x, adj = list(zip(*self.X))
        x = np.concatenate([np.expand_dims(v, axis=0) for v in x], axis=0)
        x = x.reshape(x.shape[0], -1)
        adj = np.concatenate([np.expand_dims(v, axis=0) for v in adj], axis=0)
        adj = adj.reshape(adj.shape[0], -1)
        return np.concatenate((x, adj), axis=1)

    def get_n_dim(self):
        x, adj = self.X[0]
        n_x = x.shape[0]
        for sh in x.shape[1:]:
            n_x *= sh
        n_adj = adj.shape[0]
        for sh in adj.shape[1:]:
            n_adj *= sh
        return n_x + n_adj

    def get_dataset_params(self):
        atom_type_list = [key for key, val in self.label_map.items()]
        if self.dataset_name == 'qm7':
            # atom_type_list = ['C', 'N', 'S', 'O']
            b_n_type = 4
            # b_n_squeeze = 1
            b_n_squeeze = 7
            a_n_node = 7
            a_n_type = len(atom_type_list) + 1  # 5
        elif self.dataset_name == 'qm9':
            # atom_type_list = ['C', 'N', 'O', 'F']
            b_n_type = 4
            # b_n_squeeze = 1
            b_n_squeeze = 3
            a_n_node = 9
            a_n_type = len(atom_type_list) + 1  # 5
        elif self.dataset_name == 'fishtoxi':
            # atom_type_list = ['Cl', 'C', 'O', 'N', 'Br', 'S', 'P', 'I', 'F']
            b_n_type = 4
            # b_n_squeeze = 1
            b_n_squeeze = 11
            a_n_node = 22
            a_n_type = len(atom_type_list) + 1  # 5
        elif self.dataset_name == 'freesolv':
            b_n_type = 4
            # b_n_squeeze = 1
            b_n_squeeze = 11
            a_n_node = 22
            a_n_type = len(atom_type_list) + 1  # 5
        elif self.dataset_name == 'esol':
            b_n_type = 4
            # b_n_squeeze = 1
            b_n_squeeze = 11
            a_n_node = 22
            a_n_type = len(atom_type_list) + 1  # 5
        else:
            assert False, 'unknown dataset'
        result = {'atom_type_list': atom_type_list, 'b_n_type': b_n_type, 'b_n_squeeze': b_n_squeeze,
                  'a_n_node': a_n_node, 'a_n_type': a_n_type}
        return result

    def get_nx_graphs(self, edge_to_node=False):
        function = self.create_node_labeled_only_graph if edge_to_node else self.create_node_and_edge_labeled_graph
        return self.init_graphs_mp(function)

    def init_graphs_mp(self, function):
        pool = multiprocessing.Pool()
        returns = pool.map(mp_create_graph_func, [(chunk, self.label_names, function) for chunk in
                                                  chunks(self.X, os.cpu_count())])
        pool.close()
        return list(itertools.chain.from_iterable(returns))

    def create_node_and_edge_labeled_graph(self, x, full_adj, label_names=None, i=None):
        # i: int used as the graph's name
        if label_names is None:
            label_names = self.label_names

        gr = nx.Graph(name=str(i),
                      node_labels=label_names['node_labels'],
                      node_attrs=label_names['node_attrs'],
                      edge_labels=label_names['edge_labels'],
                      edge_attrs=label_names['edge_attrs'])

        # normalise all channels except the no bond channel
        adj = np.sum(full_adj[:-1], axis=0)
        rows, cols = np.where(adj == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr.add_edges_from(edges)

        # NaN check (wrong output), return empty graph
        if len(gr.nodes) == 0 or math.isnan(x[0][0]):
            gr.clear()
            return gr

        virtual_nodes = []
        # node attributes
        for i_node in gr.nodes:
            attr_node = np.where(x[i_node] == np.max(x, axis=1)[i_node])[0][0]

            # virtual node check (normally not used if the model has been trained)
            if attr_node == x.shape[1] - 1:
                virtual_nodes.append(i_node)

            gr.nodes[i_node]['attributes'] = [str(attr_node)]
            # node_label
            for i, a_name in enumerate(gr.graph['node_labels']):
                gr.nodes[i_node][a_name] = str(attr_node)
            for i, a_name in enumerate(gr.graph['node_attrs']):
                gr.nodes[i_node][a_name] = str(attr_node)
            for edge in gr.edges(data=True):
                edge[2][label_names['edge_labels'][0]] = np.where(full_adj[:, edge[0], edge[1]])[0][0]

        for node in virtual_nodes:
            gr.remove_node(node)
        return gr

    def create_node_labeled_only_graph(self, x, full_adj, label_names=None, i=None):
        # i: int used as the graph's name
        if label_names is None:
            label_names = self.label_names

        gr = nx.Graph(name=str(i),
                      node_labels=label_names['node_labels'],
                      node_attrs=label_names['node_attrs'],
                      edge_labels=label_names['edge_labels'],
                      edge_attrs=label_names['edge_attrs'])

        # normalise all channels except the no bond channel
        adj = np.sum(full_adj[:-1], axis=0)
        rows, cols = np.where(adj == 1)
        edges = zip(rows.tolist(), cols.tolist())
        # create node for each edge
        i_node = len(x)
        n_edges = []
        edges_labels = {}
        # list to avoid the two nodes for one edge
        edge_done = []
        for row, col in edges:
            if (row, col) not in edge_done:
                edge_done += [(row, col), (col, row)]
                n_edges.append((row, i_node))
                n_edges.append((i_node, col))
                edges_labels[i_node] = x.shape[1] + np.where(full_adj[:, row, col])[0][0]
                i_node += 1

        gr.add_edges_from(n_edges)

        # NaN check (wrong output), return empty graph
        if len(gr.nodes) == 0 or math.isnan(x[0][0]):
            gr.clear()
            return gr

        virtual_nodes = []
        # node attributes
        for i_node in gr.nodes:
            # real nodes
            if i_node < x.shape[0]:
                attr_node = np.where(x[i_node] == np.max(x, axis=1)[i_node])[0][0]
            else:  # edges transformed into nodes
                attr_node = edges_labels[i_node]

            # virtual node check (normally not used if the model has been trained)
            if attr_node == x.shape[1] - 1:
                virtual_nodes.append(i_node)

            gr.nodes[i_node]['attributes'] = [str(attr_node)]
            for i, a_name in enumerate(gr.graph['node_labels']):
                gr.nodes[i_node][a_name] = str(attr_node)
            for i, a_name in enumerate(gr.graph['node_attrs']):
                gr.nodes[i_node][a_name] = str(attr_node)

        for node in virtual_nodes:
            gr.remove_node(node)
            # remove node corresponding to edges connected to the virtual node, note that we can directly remove the
            # node because nodes can only be linked to a node corresponding to an edge (in this function case)
            for (n1, n2) in n_edges:
                if n1 == node:
                    gr.remove_node(n2)
                elif n2 == node:
                    gr.remove_node(n1)
        return gr

    def __getitem__(self, idx):
        sample = self.X[idx]
        y = self.true_labels[idx]
        if self.transform:
            sample = self.transform(*sample)

        # sample = tuple([*sample, self.y[idx]])

        return sample, y

    # def init_labelmin(self):
    #     # mat = scipy.spatial.distance.cdist(np.expand_dims(self.true_labels, axis=1),
    #     #                                    np.expand_dims(self.true_labels, axis=1))
    #     # mat[np.where(mat == 0)] = 1
    #     # label_mindist = mat.min()
    #
    #     a = np.sort(self.true_labels)
    #     max_dist_neig = 0
    #     sum_dist_neig = 0
    #     for i in range(a.shape[0] - 1):
    #         sum_dist_neig += a[i + 1] - a[i]
    #         if a[i + 1] - a[i] > max_dist_neig:
    #             max_dist_neig = a[i + 1] - a[i]
    #     mean_dist_neig = sum_dist_neig / (a.shape[0] - 1)
    #
    #     return mean_dist_neig
    #     # return label_mindist

    def reduce_dataset(self, reduce_type, label=None, how_many=None, reduce_from_ori=True):
        assert self.dataset_name not in REGRESSION_DATASETS, "the dataset can\'t be reduced (regression dataset)"
        return self.reduce_dataset(reduce_type, label, how_many, reduce_from_ori)

    @staticmethod
    def load_dataset(name):
        path = './datasets'

        if name in ['qm7', 'qm9', 'freesolv', 'esol', 'lipo']:
            results, label_map = get_molecular_dataset_mp(name=name, data_path=path)
        # elif name == 'lipo':
        #
        #     from deepchem.molnet import load_lipo
        #
        #     dataset = load_lipo(featurizer='Raw')
        #     print('a')
        #
        #     ### importing OGB
        #     # from ogb.graphproppred import PygGraphPropPredDataset
        #     # dataset = PygGraphPropPredDataset(name=name)
        #     # split_idx = dataset.get_idx_split()

        else:
            # exist_dataset = os.path.exists(f'{path}/{name}_data.npy') if path is not None else False
            # test_dataset = None
            # fulldata = True
            # name_c = name
            # if fulldata and name in ['aquatoxi', 'fishtoxi']:
            #     name_c = f'{name}_graph_fulldata'
            #     exist_dataset = os.path.exists(f'{path}/{name_c}_data.npy')

            if name == 'fishtoxi':
                # if exist_dataset:
                #     X = np.load(f'{path}/{name_c}_data.npy')
                #     labels = np.load(f'{path}/{name_c}_labels.npy')
                #     if fulldata:
                #         X_test = np.load(f'{path}/{name_c}_testdata.npy')
                #         labels_test = np.load(f'{path}/{name_c}_testlabels.npy')
                #         test_dataset = (X_test, labels_test)

                name += '_graph_fulldata'
                import pandas as pd
                from rdkit import Chem
                from utils.graphs.molecular_graph_utils import get_atoms_adj_from_mol, process_mols
                mat = pd.read_excel(f'{path}/qsar_fish_toxicity_fulldata.xlsx', engine='openpyxl')
                mat = np.array(mat)

                smiles = mat[np.where(mat[:, 2] == 'train'), 1].squeeze()
                labels = mat[np.where(mat[:, 2] == 'train'), 3].squeeze().astype(np.float64)
                smiles_test = mat[np.where(mat[:, 2] == 'test'), 1].squeeze()
                y_test = mat[np.where(mat[:, 2] == 'test'), 3].squeeze().astype(np.float64)
                mols = []
                mols_test = []
                n_atoms = []
                nsmiles = []
                nsmiles_test = []
                n_labels = []
                n_y_test = []
                # filter_n_atom = 250
                filter_n_atom = 22
                for i, smile in enumerate(smiles):
                    mol = Chem.MolFromSmiles(smile)
                    if mol.GetNumAtoms() <= filter_n_atom:
                        mols.append(mol)
                        n_atoms.append(mol.GetNumAtoms())
                        nsmiles.append(Chem.MolToSmiles(mol))
                        n_labels.append(labels[i])
                for i, smile in enumerate(smiles_test):
                    mol = Chem.MolFromSmiles(smile)
                    if mol.GetNumAtoms() <= filter_n_atom:
                        mols_test.append(mol)
                        n_atoms.append(mol.GetNumAtoms())
                        nsmiles_test.append(Chem.MolToSmiles(mol))
                        n_y_test.append(y_test[i])

                # Histogram of n_atoms
                # import matplotlib.pyplot as plt
                #
                # plt.hist(n_atoms, bins=range(0, max(n_atoms)))
                # plt.savefig('hist.png')
                # plt.show()
                # plt.close()

                label_map = {}
                for mol in mols + mols_test:
                    for atom in mol.GetAtoms():
                        symbol = atom.GetSymbol()
                        if symbol not in label_map:
                            label_map[symbol] = len(label_map) + 1

                results, label_map = process_mols(name, (mols, n_labels), (mols_test, n_y_test), max(n_atoms),
                                                  label_map,
                                                  dupe_filter=False, categorical_values=False,
                                                  return_smiles=False)

                # test_dataset = (X_test, y_test)
                # np.save(f'{path}/{name}_testdata.npy', X_test)
                # np.save(f'{path}/{name}_testlabels.npy', y_test)
            else:
                assert False, 'unknown dataset'

        return results, label_map

    @staticmethod
    def format_data(input, n_bits, n_bins, device):
        for i in range(len(input)):
            input[i] = input[i].float().to(device)
        return input

    def format_loss(self, log_p, logdet, n_bins):
        loss = logdet + log_p
        return -loss.mean(), log_p.mean(), logdet.mean()

    @staticmethod
    def rescale(x):
        return x

    def is_regression_dataset(self):
        return True
