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

from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, make_moons, make_swiss_roll, \
    fetch_olivetti_faces
from utils import visualize_flow
from torch_geometric.datasets.tu_dataset import TUDataset

SIMPLE_DATASETS = ['single_moon', 'double_moon', 'iris', 'bcancer']
IMAGE_DATASETS = ['mnist', 'cifar10', 'olivetti_faces']
SIMPLE_REGRESSION_DATASETS = ['swissroll', 'diabetes', 'waterquality', 'aquatoxi', 'fishtoxi', 'trafficflow']
GRAPH_REGRESSION_DATASETS = ['qm7', 'qm9', 'freesolv', 'esol', 'lipo', 'fishtoxi']
GRAPH_CLASSIFICATION_DATASETS = ['toxcast', 'AIDS', 'Letter-low', 'Letter-med', 'Letter-high', 'MUTAG', 'COIL-DEL', 'BZR', 'BACE']


# abstract base kernel dataset class
class BaseDataset(Dataset):
    def __init__(self, X, true_labels, val_dataset=None, test_dataset=None, add_feature=None):
        self.ori_X = X
        self.ori_true_labels = true_labels
        self.X = X
        self.true_labels = true_labels

        self.idx = None

        self.reduce_type = 'all'

        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.in_size = -1

        self.add_feature = add_feature

    def __len__(self):
        return len(self.X)

    def get_flattened_X(self, with_added_features=False):
        x = self.X.reshape(self.X.shape[0], -1)
        if with_added_features and self.add_feature is not None and self.add_feature > 0:
            x_shape = [sh for sh in x.shape]
            x_shape[-1] = x_shape[-1] + self.add_feature
            n_X = np.zeros(tuple(x_shape))
            n_X[:, :-self.add_feature] = x
            x = n_X
        return x

    def get_n_dim(self, add_feature=None):
        x = self.X[0]
        n_dim = x.shape[0]
        for i in range(len(x.shape[1:])):
            sh = x.shape[i + 1]
            if i + 1 == len(x.shape[1:]):
                if add_feature is not None:
                    n_dim *= (sh + add_feature)
                elif self.add_feature is not None:
                    n_dim *= (sh + self.add_feature)
                else:
                    n_dim *= sh
            else:
                n_dim *= sh

        return n_dim

    def get_dim_per_label(self, add_feature=None, return_total_dim=False):
        n_dim = self.get_n_dim(add_feature)

        if not self.is_regression_dataset():
            uni = np.unique(self.true_labels)
            dim_per_label = math.floor(n_dim / len(uni))
        else:
            dim_per_label = n_dim

        if return_total_dim:
            return dim_per_label, n_dim
        else:
            return dim_per_label

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
            self.idx = np.array(n_idx)
        self.X = n_X
        self.true_labels = n_true_labels

    def split_dataset(self, ratio, stratified=False, split_type=None):
        # split_type -> select between (None, 'val', 'test')
        part2_dataset = copy.deepcopy(self)
        part1_dataset = copy.deepcopy(self)
        if split_type == 'test' and self.test_dataset is not None:
            print('Test dataset known, split using the test dataset...')
            part2_dataset.X = self.test_dataset[0]
            part2_dataset.true_labels = self.test_dataset[1]
            part2_dataset.val_dataset = None
            part2_dataset.test_dataset = None
            part2_dataset.idx = None

            part1_dataset.test_dataset = None
        elif split_type == 'val' and self.val_dataset is not None:
            print('Val dataset known, split using the val dataset...')
            part2_dataset.X = self.val_dataset[0]
            part2_dataset.true_labels = self.val_dataset[1]
            part2_dataset.val_dataset = None
            part2_dataset.test_dataset = None
            part2_dataset.idx = None

            part1_dataset.val_dataset = None
        else:
            if stratified:
                if self.is_regression_dataset():
                    class_sample_count = np.histogram(self.true_labels)[0]
                else:
                    class_sample_count = np.histogram(self.true_labels, bins=np.concatenate(
                        (np.unique(self.true_labels), np.ones(1) + np.max(self.true_labels))))[0]
                idxs = np.argsort(self.true_labels)
                done = 0
                val_idx = []
                for i in range(len(class_sample_count)):
                    nb_idx = math.floor(class_sample_count[i] * ratio)
                    if nb_idx == 0 and class_sample_count[i] > 1:
                        nb_idx = 1
                    val_idx += random.sample(list(idxs[done:done + class_sample_count[i]]), k=nb_idx)
                    done += class_sample_count[i]

            else:
                val_idx = random.sample(range(0, self.X.shape[0]), k=math.floor(self.X.shape[0] * ratio))

            part2_dataset.X = self.X[val_idx]
            train_idx = [idx for idx in range(self.X.shape[0]) if idx not in val_idx]
            part1_dataset.X = self.X[train_idx]
            part2_dataset.true_labels = self.true_labels[val_idx]

            part1_dataset.true_labels = self.true_labels[train_idx]

            if self.idx is None:
                part2_dataset.idx = np.array(val_idx)
                part1_dataset.idx = np.array(train_idx)
            else:
                part2_dataset.idx = self.idx[val_idx]
                part1_dataset.idx = self.idx[train_idx]

        return part1_dataset, part2_dataset

    def reduce_dataset_ratio(self, ratio, stratified=True):
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

        self.X = self.X[n_idx]
        self.true_labels = self.true_labels[n_idx]

        if self.idx is None:
            self.idx = np.array(n_idx)
        else:
            self.idx = self.idx[n_idx]

    def duplicate(self):
        return copy.deepcopy(self)

    def save_split(self, save_path):
        if self.idx is not None:
            np.save(save_path, self.idx)
        else:
            print('The dataset doesn\'t have idx to save!')

    def load_split(self, load_path, return_idx=False):
        if os.path.exists(load_path):
            self.idx = np.load(load_path)
            self.X = self.ori_X[self.idx]
            self.true_labels = self.ori_true_labels[self.idx]
            if return_idx:
                return self.idx
        else:
            assert False, f'No file to load split at the path : {load_path}'

    # To implement
    def __getitem__(self, idx):
        raise NotImplementedError

    def is_regression_dataset(self):
        raise NotImplementedError


class SimpleDataset(BaseDataset):
    def __init__(self, dataset_name, transform=None, add_feature=None):
        self.dataset_name = dataset_name
        self.label_type = np.int if self.dataset_name not in SIMPLE_REGRESSION_DATASETS else np.float
        # self.label_type = np.int
        self.transform = transform
        # ori_X, ori_true_labels, test_dataset = self.load_dataset(dataset_name, add_feature=add_feature)
        ori_X, ori_true_labels, val_dataset, test_dataset = self.load_dataset(dataset_name)
        super().__init__(ori_X, ori_true_labels, val_dataset, test_dataset, add_feature=add_feature)

        self.in_size = ori_X.shape[-1]

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
        assert self.dataset_name not in SIMPLE_REGRESSION_DATASETS, "the dataset can\'t be reduced (regression dataset)"
        return self.reduce_dataset(reduce_type, label, how_many, reduce_from_ori)

    @staticmethod
    # def load_dataset(name, add_feature=None):
    def load_dataset(name):
        path = './datasets'
        exist_dataset = os.path.exists(f'{path}/{name}_data.npy') if path is not None else False
        val_dataset = None
        test_dataset = None
        fulldata = True
        name_c = name

        if 'randgen' in name:
            n_train_data = int(name.split('-')[1])
            n_features = int(name.split('-')[2])
            n_data = int(n_train_data / 0.9)
            X = np.random.random((n_data, n_features))
            labels = (np.random.random(n_data) * 2).astype(np.int)
            return X, labels, val_dataset, test_dataset

        if fulldata and name in ['aquatoxi', 'fishtoxi']:
            name_c = f'{name}_fulldata'
            exist_dataset = os.path.exists(f'{path}/{name_c}_data.npy')
        if exist_dataset:
            # X = np.load(f'{path}/{name_c}_data.npy') /5
            X = np.load(f'{path}/{name_c}_data.npy')
            labels = np.load(f'{path}/{name_c}_labels.npy')
            # test denoising
            # labels = np.zeros_like(labels).astype(np.int)
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

        # if add_feature is not None:
        #     # TEST ADD ZERO FEATURES IN X
        #     n_X = np.zeros((X.shape[0], X.shape[-1] + add_feature))
        #     n_X[:, :-add_feature] = X
        #     X = n_X
        #
        #     if test_dataset is not None:
        #         X_test = test_dataset[0]
        #         n_X = np.zeros((X_test.shape[0], X_test.shape[-1] + add_feature))
        #         n_X[:, :-add_feature] = X_test
        #         X_test = n_X
        #         test_dataset = (X_test, test_dataset[1])

        return X, labels, val_dataset, test_dataset

    def format_data(self, input, device, add_feature=None, force_no_added_feature=False):
        x_shape = input.shape
        if not force_no_added_feature:
            if add_feature is not None and add_feature > 0:
                n_X = torch.zeros(x_shape[0], x_shape[-1] + add_feature)
                n_X[:, :-add_feature] = input
                input = n_X
            elif self.add_feature is not None and self.add_feature > 0:
                n_X = torch.zeros(x_shape[0], x_shape[-1] + self.add_feature)
                n_X[:, :-self.add_feature] = input
                input = n_X
        input = input.float().to(device)
        return input

    def format_loss(self, log_p, logdet):
        loss = logdet + log_p
        return -loss.mean(), log_p.mean(), logdet.mean()

    @staticmethod
    def rescale(x):
        return x

    def is_regression_dataset(self):
        # return False
        return self.dataset_name in SIMPLE_REGRESSION_DATASETS


class ImDataset(BaseDataset):
    def __init__(self, dataset_name, transform=None, noise_transform=None, n_bits=5, add_feature=None):
        self.dataset_name = dataset_name
        train_dataset, val_dataset, test_dataset = ImDataset.load_dataset(dataset_name, transform)
        self.transform = train_dataset.transform
        if self.transform is not None:
            for tr in self.transform.transforms:
                if isinstance(tr, transforms.Normalize):
                    self.norm_mean = np.array(tr.mean).reshape(1, -1, 1, 1)
                    self.norm_std = np.array(tr.std).reshape(1, -1, 1, 1)
                    self.rescale = self.rescale_val_to_im_with_norm
        if not hasattr(self, 'norm_mean'):
            self.rescale = self.rescale_val_to_im_without_norm

        def get_X_y_from_dset(dataset):
            if isinstance(train_dataset.data, np.ndarray):
                X = train_dataset.data
            else:
                X = train_dataset.data.numpy()
            if isinstance(train_dataset.targets, np.ndarray):
                true_labels = train_dataset.targets
            else:
                true_labels = np.array(train_dataset.targets)
            return X, true_labels

        ori_X, ori_true_labels = get_X_y_from_dset(train_dataset)
        if test_dataset is not None:
            test_ori_X, test_true_labels = get_X_y_from_dset(test_dataset)
            test_dataset = (test_ori_X, test_true_labels)
        if val_dataset is not None:
            val_ori_X, val_true_labels = get_X_y_from_dset(val_dataset)
            val_dataset = (val_ori_X, val_true_labels)

        super().__init__(ori_X, ori_true_labels, val_dataset, test_dataset, add_feature)
        print('Z and K are not initialized in constructor')

        # test with denoise loss
        self.noise_transform = noise_transform
        self.n_channel = train_dataset.data.shape[1]
        self.in_size = train_dataset.data.shape[-1]

        self.current_mode = 'XY'
        self.get_func = self.get_XY_item

        self.get_PIL_image = self.get_PIL_image_L if self.n_channel == 1 else self.get_PIL_image_RGB
        if self.dataset_name == 'olivetti_faces':
            self.get_PIL_image = self.get_PIL_image_1b
        self.n_bits = n_bits
        self.n_bins = 2.0 ** n_bits

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

    def get_PIL_image_1b(self, idx):
        im = self.X[idx]
        im = Image.fromarray(im.squeeze())
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

    def format_data(self, input, device, add_feature=None, force_no_added_feature=False):
        input = input * 255

        if self.n_bits < 8:
            input = torch.floor(input / 2 ** (8 - self.n_bits))

        input = input / self.n_bins - 0.5
        input = input + torch.rand_like(input) / self.n_bins

        x_shape = input.shape
        if not force_no_added_feature:
            if add_feature is not None and add_feature > 0:
                n_X = torch.zeros(x_shape[0], x_shape[1], x_shape[-1] + add_feature)
                n_X[:, :, :-add_feature] = input
                input = n_X
            elif self.add_feature is not None and self.add_feature > 0:
                n_X = torch.zeros(x_shape[0], x_shape[1], x_shape[-1] + self.add_feature)
                n_X[:, :, :-self.add_feature] = input
                input = n_X
        input = input.to(device)
        return input

    def format_loss(self, log_p, logdet):
        n_pixel = self.in_size * self.in_size * 3

        loss = -math.log(self.n_bins) * n_pixel
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
    def load_dataset(name, transform=None):
        test_dataset = None
        val_dataset = None

        class WrappedDataset:
            def __init__(self, X, labels, transform):
                self.data = X
                self.targets = labels
                self.transform = transform

        if 'randgen' in name:
            transform = transforms.Compose(
                [transforms.ToTensor()]
            )
            n_train_data = int(name.split('-')[1])
            n_features = int(name.split('-')[2])
            n_data = int(n_train_data / 0.9)
            n_features_dim = math.floor(math.sqrt(n_features) / 4) * 4  # each dim should be div by 4 (2blocks)
            print(f'Random dataset features_dim ={n_features_dim}x{n_features_dim}')
            X = np.random.random((n_data, 1, n_features_dim, n_features_dim))
            labels = (np.random.random(n_data) * 2).astype(np.int)
            train_dataset = WrappedDataset(X, labels, transform)
            return train_dataset, val_dataset, test_dataset

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
            train_dataset = datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
            train_dataset.data = train_dataset.data.unsqueeze(1)
            test_dataset.data = test_dataset.data.unsqueeze(1)

            # keep only 1 label
            # train_dataset.data = train_dataset.data[train_dataset.targets == 3]
            # train_dataset.targets = np.zeros(train_dataset.data.shape[0]).astype(np.int)
            # test_dataset.data = test_dataset.data[test_dataset.targets == 3]
            # test_dataset.targets = np.zeros(test_dataset.data.shape[0]).astype(np.int)
        elif name == 'cifar10':
            if transform is None:
                transform = transforms.Compose([
                    # transforms.RandomCrop(32, padding=4),
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

            train_dataset = datasets.CIFAR10('./datasets', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
            train_dataset.data = train_dataset.data.transpose(0, 3, 1, 2)
            test_dataset.data = test_dataset.data.transpose(0, 3, 1, 2)
        elif name == 'olivetti_faces':
            if transform is None:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])
            X, labels = fetch_olivetti_faces(return_X_y=True)
            X = X.reshape(X.shape[0], 1, np.sqrt(X.shape[-1]).astype(int), np.sqrt(X.shape[-1]).astype(int))
            train_dataset = WrappedDataset(X, labels, transform)
        else:
            assert False, 'unknown dataset'

        return train_dataset, val_dataset, test_dataset

    def is_regression_dataset(self):
        return self.dataset_name in SIMPLE_REGRESSION_DATASETS


import networkx as nx
from utils.graphs.files_manager import DataLoader
from utils.graphs.molecular_graph_utils import get_molecular_dataset_mp
from utils.graphs.utils_datasets import transform_graph_permutation
import multiprocessing
import itertools

ATOMIC_NUM_MAP = {'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53, 'Si': 14, 'Ba': 56,
                  'Nd': 60, 'Dy': 66, 'In': 49, 'Sb': 51, 'Co': 27, 'B': 5, 'Ca': 20, 'Ni': 28, 'Na': 11, 'Se': 34,
                  'Tl': 81, 'Cd': 48, 'Yb': 70, 'Li': 3, 'Cr': 24, 'Sn': 50, 'Hg': 80, 'Mn': 25, 'K': 19, 'As': 33,
                  'Bi': 83, 'Mo': 42, 'Mg': 12, 'Au': 79, 'Zr': 40, 'Zn': 30, 'Cu': 29, 'Fe': 26, 'Eu': 63, 'Al': 13,
                  'Pt': 78, 'Sr': 38, 'Sc': 21, 'Ti': 22, 'Ag': 47, 'Pb': 82, 'Pd': 46, 'Be': 4, 'Ge': 32,
                  'V': 23, 'Gd': 64, 'Ru': 44, 'Te': 52, 'W': 74, 'Rh': 45, 'Ga': 31, 'Ho': 67, 'Tb': 65}


def chunks(lst, n):
    step = int(len(lst) / n) if n < len(lst) else 1
    for i in range(0, len(lst), step):
        yield lst[i:i + step]


def mp_create_graph_func(input_args):
    # input_args is a tuple of ([(x,adj),(x2,adj2),...],label_names dict, create_graph_function)
    chunk, label_names, create_graph_fun, attributed_node = input_args
    return [create_graph_fun(*chunk[i], label_names=label_names, i=i, attributed_node=attributed_node) for i in
            range(len(chunk))]


class GraphDataset(BaseDataset):
    def __init__(self, dataset_name, transform=None, add_feature=None):
        self.dataset_name = dataset_name
        # self.label_type = np.int if self.dataset_name not in REGRESSION_DATASETS else np.float

        (train_dataset, val_dataset, test_dataset), self.label_map = self.load_dataset(
            dataset_name)  # , add_feature=add_feature)
        if len(train_dataset) == 3:
            xs, adjs, y = train_dataset
        else:
            xs, adjs, self.smiles, y = train_dataset

        y = np.array(y).squeeze() if self.is_regression_dataset() else np.array(y).astype(np.int).squeeze()

        def format_dset(dataset):
            if len(dataset) == 3:
                xs, adjs, y = dataset
            else:
                xs, adjs, smiles, y = dataset
            y = np.array(y).squeeze() if self.is_regression_dataset() else np.array(y).astype(np.int).squeeze()
            return list(zip(xs, adjs)), np.array(y).squeeze()

        if test_dataset is not None:
            test_dataset = format_dset(test_dataset)
        if val_dataset is not None:
            val_dataset = format_dset(val_dataset)

        self.transform = transform_graph_permutation if transform == 'permutation' else None
        if self.label_map is not None:
            self.atomic_num_list = [ATOMIC_NUM_MAP[label] for label, _ in self.label_map.items()] + [0]
        else:
            print('Not a molecule dataset -> no atom_num conversion.')
            self.atomic_num_list = None
        graphs = list(zip(xs, adjs))

        # Graphs for tests
        self.define_networkx_labels()

        super().__init__(graphs, y, val_dataset, test_dataset, add_feature=add_feature)
        # self.Gn = self.init_graphs_mp()

        # if self.is_regression_dataset():
        #     self.label_mindist = self.init_labelmin()

    def get_input_shapes(self):
        x_shape = list(self.X[0][0].shape)
        if self.add_feature is not None:
            x_shape[-1] += self.add_feature
        x_shape = tuple(x_shape)
        adj_shape = self.X[0][1].shape
        return x_shape, adj_shape

    def is_attributed_node_dataset(self):
        return 'Letter' in self.dataset_name #or self.dataset_name in ['Letter-med']

    def define_networkx_labels(self):
        self.label_names = {'node_labels': ['node_attr'], 'node_attrs': ['node_attr'], 'edge_labels': ['bond_attr'],
                            'edge_attrs': ['bond_attr']}
        self.node_labels = self.label_names['node_labels']
        self.node_attrs = self.label_names['node_attrs']
        self.edge_labels = self.label_names['edge_labels']
        self.edge_attrs = self.label_names['edge_attrs']

    # overwrite
    def reduce_dataset(self, reduce_type, label=None, how_many=None, reduce_from_ori=True):
        raise NotImplementedError

    # overwrite
    def split_dataset(self, ratio, stratified=False, split_type=None):
        # split_type -> select between (None, 'val', 'test')
        part2_dataset = copy.deepcopy(self)
        part1_dataset = copy.deepcopy(self)
        if split_type == 'test' and self.test_dataset is not None:
            print('Test dataset known, split using the test dataset...')
            part2_dataset.X = self.test_dataset[0]
            part2_dataset.true_labels = self.test_dataset[1]
            part2_dataset.val_dataset = None
            part2_dataset.test_dataset = None
            part2_dataset.idx = None

            part1_dataset.test_dataset = None
        elif split_type == 'val' and self.val_dataset is not None:
            print('Val dataset known, split using the val dataset...')
            part2_dataset.X = self.val_dataset[0]
            part2_dataset.true_labels = self.val_dataset[1]
            part2_dataset.val_dataset = None
            part2_dataset.test_dataset = None
            part2_dataset.idx = None

            part1_dataset.val_dataset = None
        else:
            if stratified:
                if self.is_regression_dataset():
                    class_sample_count = np.histogram(self.true_labels)[0]
                else:
                    class_sample_count = np.histogram(self.true_labels, bins=np.concatenate(
                        (np.unique(self.true_labels), np.ones(1) + np.max(self.true_labels))))[0]
                idxs = np.argsort(self.true_labels)
                done = 0
                val_idx = []
                for i in range(len(class_sample_count)):
                    nb_idx = math.floor(class_sample_count[i] * ratio)
                    if nb_idx == 0 and class_sample_count[i] > 1:
                        nb_idx = 1
                    val_idx += random.sample(list(idxs[done:done + class_sample_count[i]]), k=nb_idx)
                    done += class_sample_count[i]
            else:
                val_idx = random.sample(range(0, len(self.X)), k=math.floor(len(self.X) * ratio))

            train_idx = [idx for idx in range(len(self.X)) if idx not in val_idx]
            x, adj = list(zip(*self.X))
            x = np.array(x)
            adj = np.array(adj)
            val_x = x[val_idx]
            val_adj = adj[val_idx]
            val_X = []
            for i in range(len(val_idx)):
                val_X.append((val_x[i], val_adj[i]))
            part2_dataset.X = val_X
            train_x = x[train_idx]
            train_adj = adj[train_idx]
            train_X = []
            for i in range(len(train_idx)):
                train_X.append((train_x[i], train_adj[i]))
            part1_dataset.X = train_X

            part2_dataset.true_labels = self.true_labels[val_idx]
            part1_dataset.true_labels = self.true_labels[train_idx]

            if self.idx is None:
                part2_dataset.idx = np.array(val_idx)
                part1_dataset.idx = np.array(train_idx)
            else:
                part2_dataset.idx = self.idx[val_idx]
                part1_dataset.idx = self.idx[train_idx]

        return part1_dataset, part2_dataset

    # overwrite
    def reduce_dataset_ratio(self, ratio, stratified=True):
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
            n_idx = random.sample(range(0, len(self.X)), k=math.floor(len(self.X) * ratio))

        x, adj = list(zip(*self.X))
        x = np.array(x)
        adj = np.array(adj)
        n_x = x[n_idx]
        n_adj = adj[n_idx]
        n_X = []
        for i in range(len(n_idx)):
            n_X.append((n_x[i], n_adj[i]))
        self.X = n_X
        self.true_labels = self.true_labels[n_idx]

        if self.idx is None:
            self.idx = np.array(n_idx)
        else:
            self.idx = np.array(self.idx)[n_idx].tolist()

    # overwrite
    def load_split(self, load_path, return_idx=False):
        if os.path.exists(load_path):
            self.idx = np.load(load_path)
            x, adj = list(zip(*self.ori_X))
            x = np.array(x)
            adj = np.array(adj)
            train_x = x[self.idx]
            train_adj = adj[self.idx]
            train_X = []
            for i in range(len(self.idx)):
                train_X.append((train_x[i], train_adj[i]))
            self.X = train_X
            self.true_labels = self.ori_true_labels[self.idx]
            if return_idx:
                return self.idx
        else:
            assert False, f'No file to load split at the path : {load_path}'

    def permute_graphs_in_dataset(self):
        for i in range(len(self.X)):
            self.X[i] = transform_graph_permutation(*self.X[i])

    def get_flattened_X(self, with_added_features=False):
        x, adj = list(zip(*self.X))
        x = np.concatenate([np.expand_dims(v, axis=0) for v in x], axis=0)
        if with_added_features and self.add_feature is not None and self.add_feature > 0:
            n_X = np.zeros((x.shape[0], x.shape[1], x.shape[-1] + self.add_feature))
            n_X[:, :, :-self.add_feature] = x
            x = n_X
        x = x.reshape(x.shape[0], -1)
        adj = np.concatenate([np.expand_dims(v, axis=0) for v in adj], axis=0)
        adj = adj.reshape(adj.shape[0], -1)
        return np.concatenate((x, adj), axis=1)

    def calculate_dims(self, add_feature=None):
        x, adj = self.X[0]
        n_x = x.shape[0]
        for i in range(len(x.shape[1:])):
            sh = x.shape[i + 1]
            if i + 1 == len(x.shape[1:]):
                if add_feature is not None:
                    n_x *= (sh + add_feature)
                elif self.add_feature is not None:
                    n_x *= (sh + self.add_feature)
                else:
                    n_x *= sh
            else:
                n_x *= sh
        n_adj = adj.shape[0]
        for sh in adj.shape[1:]:
            n_adj *= sh
        return n_x, n_adj

    def get_n_dim(self, add_feature=None):
        n_x, n_adj = self.calculate_dims(add_feature=add_feature)
        return n_x + n_adj

    def get_nx_graphs(self, edge_to_node=False, data=None, attributed_node=False):
        function = self.create_node_labeled_only_graph if edge_to_node else self.create_node_and_edge_labeled_graph
        return self.init_graphs_mp(function, data=data, attributed_node=attributed_node)

    def init_graphs_mp(self, function, data=None, attributed_node=False):
        data = data if data is not None else self.X
        # pool = multiprocessing.Pool()
        # returns = pool.map(mp_create_graph_func, [(chunk, self.label_names, function, attributed_node) for chunk in
        #                                           chunks(data, os.cpu_count())])
        # pool.close()
        returns = [function(*data[i], label_names=self.label_names, i=i, attributed_node=attributed_node)
                   for i in range(len(data))]
        return returns
        # return list(itertools.chain.from_iterable(returns))

    def create_node_and_edge_labeled_graph(self, x, full_adj, label_names=None, i=None, attributed_node=False):
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

        # case 1 node, no edge
        if len(gr.nodes) == 0 and np.unique(x, axis=0).shape[0] > 1:
            for i, x_i in enumerate(x):
                if (x_i != np.zeros(x_i.shape)).all():
                    gr.add_node(i)

        # NaN check (wrong output), return empty graph
        if len(gr.nodes) == 0 or math.isnan(x[0][0]):
            gr.clear()
            return gr

        virtual_nodes = []
        # node attributes
        for i_node in gr.nodes:
            if not attributed_node:
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
            else:
                attr_node = x[i_node]

                gr.nodes[i_node]['node_attr'] = tuple(attr_node)
                # node_label
                # for i, a_name in enumerate(gr.graph['node_attrs']):
                #     gr.nodes[i_node][a_name] = attr_node[i]
                for edge in gr.edges(data=True):
                    edge[2][label_names['edge_labels'][0]] = np.where(full_adj[:, edge[0], edge[1]])[0][0]

                # virtual node check
                # if (attr_node == np.zeros_like(attr_node)).all():
                # if attr_node[-1] > 0.5:
                #     virtual_nodes.append(i_node)

        for node in virtual_nodes:
            gr.remove_node(node)
        return gr

    def create_node_labeled_only_graph(self, x, full_adj, label_names=None, i=None, attributed_node=False):
        assert not attributed_node, 'node only graphs creation (edge -> node) support only labeled node/edge,' \
                                    ' not attributed'
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

        removed_edge = []
        for node in virtual_nodes:
            gr.remove_node(node)
            # remove node corresponding to edges connected to the virtual node, note that we can directly remove the
            # node because nodes can only be linked to a node corresponding to an edge (in this function case)
            for (n1, n2) in n_edges:
                if n1 == node and n2 not in removed_edge:
                    removed_edge.append(n2)
                    gr.remove_node(n2)
                elif n2 == node and n1 not in removed_edge:
                    removed_edge.append(n1)
                    gr.remove_node(n1)
        return gr

    def get_full_graphs(self, data=None, attributed_node=False):
        return self.full_graphs_mp(data=data, attributed_node=attributed_node)

    def full_graphs_mp(self, data=None, attributed_node=False):
        data = data if data is not None else self.X
        # pool = multiprocessing.Pool()
        # returns = pool.map(mp_create_graph_func,
        #                    [(chunk, self.label_names, self.create_full_graph, attributed_node) for chunk in
        #                     chunks(data, os.cpu_count())])
        # pool.close()
        returns = [self.create_full_graph(*data[i], label_names=self.label_names, i=i, attributed_node=attributed_node)
                   for i in range(len(data))]
        return returns
        # return list(itertools.chain.from_iterable(returns))

    def create_full_graph(self, x, full_adj, label_names=None, i=None, attributed_node=False):
        # i: int used as the graph's name
        if label_names is None:
            label_names = self.label_names

        gr = nx.Graph(name=str(i),
                      node_labels=label_names['node_labels'],
                      node_attrs=label_names['node_attrs'],
                      edge_labels=label_names['edge_labels'],
                      edge_attrs=label_names['edge_attrs'])

        adj = np.sum(full_adj, axis=0)
        rows, cols = np.where(adj == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr.add_edges_from(edges)

        # node attributes
        for i_node in gr.nodes:
            # If NaN only in the row
            if np.isnan(np.nanmax(x[i_node])):
                return None
            if not attributed_node:

                attr_node = np.where(x[i_node] == np.nanmax(x[i_node]))[0][0]

                gr.nodes[i_node]['attributes'] = [str(attr_node)]
                # node_label
                for i, a_name in enumerate(gr.graph['node_labels']):
                    gr.nodes[i_node][a_name] = str(attr_node)
                for i, a_name in enumerate(gr.graph['node_attrs']):
                    gr.nodes[i_node][a_name] = str(attr_node)
                for edge in gr.edges(data=True):
                    edge[2][label_names['edge_labels'][0]] = np.where(full_adj[:, edge[0], edge[1]])[0][0]
            else:
                # attr_node = x[i_node][:-1]
                attr_node = x[i_node]

                gr.nodes[i_node]['node_attr'] = tuple(attr_node)
                # node_label
                # for i, a_name in enumerate(gr.graph['node_attrs']):
                #     gr.nodes[i_node][a_name] = attr_node[i]
                for edge in gr.edges(data=True):
                    edge[2][label_names['edge_labels'][0]] = np.where(full_adj[:, edge[0], edge[1]])[0][0]

        edge_labels = nx.get_edge_attributes(gr, "bond_attr")
        # edges_no_virtual = {}
        for k, v in edge_labels.items():
            if v == full_adj.shape[0] - 1:
                gr.remove_edge(*k)
        return gr

    def __getitem__(self, idx):
        sample = self.X[idx]
        y = self.true_labels[idx]
        if self.transform:
            sample = self.transform(*sample)

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

    def format_data(self, input, device, add_feature=None, force_no_added_feature=False):
        for i in range(len(input)):
            if i == 0 and not force_no_added_feature:
                if add_feature is not None and add_feature > 0:
                    n_X = torch.zeros(input[i].shape[0], input[i].shape[1], input[i].shape[-1] + add_feature)
                    n_X[:, :, :-add_feature] = input[i]
                    input[i] = n_X
                elif self.add_feature is not None and self.add_feature > 0:
                    n_X = torch.zeros(input[i].shape[0], input[i].shape[1], input[i].shape[-1] + self.add_feature)
                    n_X[:, :, :-self.add_feature] = input[i]
                    input[i] = n_X
            input[i] = input[i].float().to(device)
        return input

    def format_loss(self, log_p, logdet):
        loss = logdet + log_p
        return -loss.mean(), log_p.mean(), logdet.mean()

    @staticmethod
    def rescale(x):
        return x

    def get_dataset_params(self, add_feature=None):
        raise NotImplementedError

    @staticmethod
    # def load_dataset(name, add_feature=None):
    def load_dataset(name):
        raise NotImplementedError

    def is_regression_dataset(self):
        raise NotImplementedError


class RegressionGraphDataset(GraphDataset):
    def __init__(self, dataset_name, transform=None, add_feature=None):
        super().__init__(dataset_name, transform=transform, add_feature=add_feature)

    def get_dataset_params(self, add_feature=None):
        # atom_type_list = [key for key, val in self.label_map.items()]
        add_f = add_feature if add_feature is not None else self.add_feature if self.add_feature is not None else None
        dim = self.X[0][0].shape[-1] - 1 if add_f is None else (self.X[0][0].shape[-1] + add_f) - 1
        atom_type_list = [i for i in range(dim)]  # -1 because of virtual node
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
            # b_n_squeeze = 12
            a_n_node = 22
            # a_n_node = 24
            a_n_type = len(atom_type_list) + 1  # 5
        elif self.dataset_name == 'esol':
            b_n_type = 4
            # b_n_squeeze = 1
            b_n_squeeze = 11
            a_n_node = 22
            a_n_type = len(atom_type_list) + 1  # 5
        elif self.dataset_name == 'lipo':
            b_n_type = 4
            # b_n_squeeze = 1
            b_n_squeeze = 5
            a_n_node = 35
            a_n_type = len(atom_type_list) + 1
        else:
            assert False, 'unknown dataset'
        result = {'atom_type_list': atom_type_list, 'b_n_type': b_n_type, 'b_n_squeeze': b_n_squeeze,
                  'a_n_node': a_n_node, 'a_n_type': a_n_type}
        return result

    @staticmethod
    # def load_dataset(name, add_feature=None):
    def load_dataset(name):
        path = './datasets'

        if name in ['qm7', 'qm9', 'freesolv', 'esol', 'lipo']:
            results, label_map = get_molecular_dataset_mp(name=name, data_path=path)
        else:
            if name == 'fishtoxi':
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

                datas = ((mols, n_labels), (mols_test, n_y_test))
                results, label_map = process_mols(name, datas, max(n_atoms), label_map,
                                                  dupe_filter=False, categorical_values=False,
                                                  return_smiles=False)

                results = (results[0], None, results[1])  # no val dataset

                # test_dataset = (X_test, y_test)
                # np.save(f'{path}/{name}_testdata.npy', X_test)
                # np.save(f'{path}/{name}_testlabels.npy', y_test)
            else:
                assert False, 'unknown dataset'

        # if add_feature is not None:
        #     if len(results[0]) == 3:
        #         ((X, A, Y), (X_test, A_test, Y_test)) = results
        #     else:
        #         ((X, A, smiles, Y), (X_test, A_test, smiles_test, Y_test)) = results
        #     X = np.concatenate([np.expand_dims(x, 0) for x in X], 0)
        #     # TEST ADD ZERO FEATURES IN X
        #     n_X = np.zeros((X.shape[0], X.shape[1], X.shape[-1] + add_feature))
        #     n_X[:, :, :-add_feature] = X
        #     X = n_X
        #
        #     X_test = np.concatenate([np.expand_dims(x, 0) for x in X_test], 0)
        #     n_X = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[-1] + add_feature))
        #     n_X[:, :, :-add_feature] = X_test
        #     X_test = n_X
        #
        #     if len(results[0]) == 3:
        #         results = ((X, A, Y), (X_test, A_test, Y_test))
        #     else:
        #         results = ((X, A, smiles, Y), (X_test, A_test, smiles_test, Y_test))

        return results, label_map

    def is_regression_dataset(self):
        return True

    # overwrite
    def reduce_dataset(self, reduce_type, label=None, how_many=None, reduce_from_ori=True):
        assert False, "the dataset can\'t be reduced (regression dataset)"


class ClassificationGraphDataset(GraphDataset):

    def __init__(self, dataset_name, transform=None, add_feature=None):
        super().__init__(dataset_name, transform=transform, add_feature=add_feature)

    def get_dataset_params(self, add_feature=None):
        # if self.label_map is not None:
        #     node_type_list = [key for key, val in self.label_map.items()]
        # else:
        add_f = add_feature if add_feature is not None else self.add_feature if self.add_feature is not None else None
        dim = self.X[0][0].shape[-1] - 1 if add_f is None else (self.X[0][0].shape[-1] + add_f) - 1
        node_type_list = [i for i in range(dim)]  # -1 because of virtual node
        if self.dataset_name == 'toxcast':
            b_n_type = 4
            b_n_squeeze = 10
            a_n_node = 30
            a_n_type = len(node_type_list) + 1  # 5
        elif self.dataset_name == 'AIDS':
            b_n_type = 4
            # b_n_squeeze = 13
            b_n_squeeze = 19
            # a_n_node = 13
            a_n_node = 95
            a_n_type = len(node_type_list) + 1  # 5
        elif self.dataset_name == 'BZR':
            b_n_type = 4
            # b_n_squeeze = 13
            b_n_squeeze = 57
            # a_n_node = 13
            a_n_node = 57
            a_n_type = len(node_type_list) + 1  # 5
        elif self.dataset_name == 'Letter-low':
            b_n_type = 2
            b_n_squeeze = 4
            a_n_node = 8
            a_n_type = len(node_type_list) + 1  # 5
        elif self.dataset_name == 'Letter-med':
            b_n_type = 2
            b_n_squeeze = 3
            a_n_node = 9
            a_n_type = len(node_type_list) + 1  # 5
        elif self.dataset_name == 'Letter-high':
            b_n_type = 2
            b_n_squeeze = 3
            a_n_node = 9
            a_n_type = len(node_type_list) + 1  # 5
        elif self.dataset_name == 'MUTAG':
            # b_n_type = 5
            b_n_type = 4
            b_n_squeeze = 14
            a_n_node = 28
            a_n_type = len(node_type_list) + 1  # 5
        elif self.dataset_name == 'COIL-DEL':
            # b_n_type = 5
            b_n_type = 3
            b_n_squeeze = 30
            a_n_node = 60
            a_n_type = len(node_type_list) + 1  # 5
        elif self.dataset_name == 'BACE':
            # b_n_type = 5
            b_n_type = 4
            b_n_squeeze = 5
            a_n_node = 50
            a_n_type = len(node_type_list) + 1  # 5
        else:
            assert False, 'unknown dataset'
        result = {'atom_type_list': node_type_list, 'b_n_type': b_n_type, 'b_n_squeeze': b_n_squeeze,
                  'a_n_node': a_n_node, 'a_n_type': a_n_type}
        return result

    @staticmethod
    def convert_tud_dataset(dataset, node_features, filter_n_nodes=13):
        n_nodes = []
        idxs = []
        filter_n_nodes = filter_n_nodes if filter_n_nodes is not None else math.inf
        max_num_node = 0
        for i, graph in enumerate(dataset):
            if graph.num_nodes <= filter_n_nodes:
                idxs.append(i)
                n_nodes.append(graph.num_nodes)
                if graph.num_nodes > max_num_node:
                    max_num_node = graph.num_nodes
        dataset_size = len(n_nodes)
        # hist = np.histogram(n_nodes, bins=range(0, max_num_node + 1))
        num_edge_features = dataset.num_edge_features if dataset.num_edge_features > 0 else 1
        num_node_features = dataset.num_node_features if dataset.num_node_features > 0 else 1

        x_feature_shape = num_node_features if node_features else num_node_features + 1  # if label +1 for virtual node
        X = np.zeros((dataset_size, max_num_node, x_feature_shape))
        A = np.zeros((dataset_size, num_edge_features + 1, max_num_node, max_num_node))
        A[:, -1, :, :] = 1
        Y = np.zeros((dataset_size)).astype(np.int)

        for ig in range(dataset_size):
            graph = dataset[idxs[ig]]
            n_node = graph.num_nodes
            X[ig, :n_node, :num_node_features] = graph.x if graph.x.shape[-1] > 0 else torch.ones(graph.x.shape[0], 1)
            # virtual nodes if label node
            if not node_features:
                X[ig, n_node:, -1] = 1
            edge_index = graph.edge_index.detach().cpu().numpy().transpose()
            edge_attr = graph.edge_attr if graph.edge_attr is not None else torch.ones(edge_index.shape[0])
            for n, (i, j) in enumerate(edge_index):
                A[ig, :num_edge_features, i, j] = edge_attr[n]
                # virtual edges
                A[ig, -1, i, j] = 0
            Y[ig] = graph.y
        return X, A, Y

    @staticmethod
    def get_filter_size(dataset_name):
        if dataset_name == 'AIDS':
            # return 22
            return None
        elif dataset_name == 'MUTAG':
            return None
        elif 'Letter' in dataset_name:
            return None
        elif dataset_name == 'COIL-DEL':
            return 60
        else:
            return None

    @staticmethod
    def clear_aromatic_molecule_bonds_from_dataset(X, A, label_map):
        import rdkit.Chem as Chem
        from utils.graphs.mol_utils import construct_mol
        from utils.graphs.molecular_graph_utils import get_atoms_adj_from_mol, atoms_to_one_hot
        virtual_bond_idx = 4
        custom_bond_assignement = [Chem.rdchem.BondType.AROMATIC, Chem.rdchem.BondType.SINGLE,
                                   Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]

        atomic_num_list = [ATOMIC_NUM_MAP[label] for label, _ in label_map.items()] + [0]
        max_num_nodes = X.shape[1]
        n_X = np.zeros_like(X)
        n_A = np.zeros((A.shape[0], A.shape[1] - 1, *A.shape[2:]))
        for i, (x, adj) in enumerate(zip(X, A)):
            mol = construct_mol(x, adj, atomic_num_list, custom_bond_assignement=custom_bond_assignement,
                                virtual_bond_idx=virtual_bond_idx)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            nx, nadj, _ = get_atoms_adj_from_mol(mol, max_num_nodes=max_num_nodes, label_map=label_map)
            nx = atoms_to_one_hot(nx, label_map)
            n_X[i, :] = nx
            n_A[i, :] = nadj
        return n_X, n_A

    @staticmethod
    # def load_dataset(name, add_feature=None):
    def load_dataset(name):
        path = './datasets'
        test_dataset = None
        val_dataset = None

        if name in ['toxcast', 'BACE']:
            results, label_map = get_molecular_dataset_mp(name=name, data_path=path, return_smiles=False)

            # ((X, A, Y), val_dataset, test_dataset) = results

            # TO TEST DATASETS
            # tasks, (train_dataset, valid_dataset, test_dataset), transformers = dc.molnet.load_toxcast(featurizer='Raw')
            # n_atoms = []
            # label_map = {}
            # for mol in train_dataset.X:
            #     n_atoms.append(mol.GetNumAtoms())
            #     for atom in mol.GetAtoms():
            #         if atom.GetSymbol() not in label_map:
            #             label_map[atom.GetSymbol()] = len(label_map) + 1
            # hist = np.histogram(n_atoms, bins=range(0, max(n_atoms) + 1))
        else:
            # full_name = f'{name}_p{add_feature}f' if add_feature is not None else name
            full_name = name

            exist_dataset = os.path.exists(f'{path}/{full_name}_X.npy') if path is not None else False
            # dset = TUDataset(path, name='DBLP_v1', use_node_attr=False, use_edge_attr=True)

            # from gklearn.utils.graph_files import load_gxl
            # import glob
            # path = f'{path}/Letter/Letter/MED'
            # files = sorted(glob.glob(f"{path}/*.gxl"))
            # graphs = []
            # for i, file in enumerate(files):
            #     graph = load_gxl(file)
            #     graphs.append(graph)
            #
            #     save_path = f'{path}/test_{i}'
            #
            #     G = graph[0]
            #     e0 = [(u, v) for (u, v, d) in G.edges(data=True)]
            #
            #     nodes = [i for i, n in enumerate(G.nodes)]
            #     nodes_pos = {}
            #     for i, n in enumerate(G.nodes):
            #         x = G.nodes[i]['x']
            #         y = G.nodes[i]['y']
            #         nodes_pos[i] = (float(x), float(y))
            #
            #     options_node = {
            #         "node_color": "skyblue",
            #     }
            #     options_edge = {
            #         "edge_color": [0 for _ in range(len(e0))],
            #         "width": 4,
            #         "edge_cmap": plt.cm.Blues_r,
            #     }
            #     # nodes
            #     nx.draw_networkx_nodes(G, nodes_pos, nodelist=nodes, **options_node)
            #
            #     # edges
            #     nx.draw_networkx_edges(G, nodes_pos, edgelist=e0, **options_edge)
            #
            #     plt.axis("off")
            #     plt.tight_layout()
            #     plt.savefig(fname=f'{save_path}.png', format='png', dpi=30)
            #     plt.close()

            if exist_dataset:
                if 'Letter' in name or name in ['AIDS', 'MUTAG', 'COIL-DEL', 'BZR']:
                    X = np.load(f'{path}/{full_name}_X.npy')
                    A = np.load(f'{path}/{full_name}_A.npy')
                    Y = np.load(f'{path}/{full_name}_Y.npy')
                    dataset = (X, A, Y)
                    results = (dataset, val_dataset, test_dataset)
                    if name == 'AIDS':
                        ordered_labels = ['C', 'O', 'N', 'Cl', 'F', 'S', 'Se', 'P', 'Na', 'I', 'Co', 'Br', 'Li', 'Si',
                                          'Mg', 'Cu', 'As', 'B', 'Pt', 'Ru', 'K', 'Pd', 'Au', 'Te', 'W', 'Rh', 'Zn',
                                          'Bi', 'Pb', 'Ge', 'Sb', 'Sn', 'Ga', 'Hg', 'Ho', 'Tl', 'Ni', 'Tb']
                        label_map = {label: i + 1 for i, label in enumerate(ordered_labels)}
                    elif name == 'MUTAG':
                        ordered_labels = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
                        label_map = {label: i + 1 for i, label in enumerate(ordered_labels)}
                    else:
                        label_map = None
                else:
                    assert False, 'unknown dataset'
            else:
                if 'Letter' in name or name in ['AIDS', 'MUTAG', 'COIL-DEL', 'BZR']:
                    # node_features = name in ['Letter-med', 'COIL-DEL']  # features if not node labels (e.g Letter-med (x,y))
                    node_features = 'Letter' in name or name in [
                                             'COIL-DEL']  # features if not node labels (e.g Letter-med (x,y))
                    dset = TUDataset(path, name=name, use_node_attr=node_features, use_edge_attr=True)
                    # TEST with virtual node
                    # node_features = False if name in ['Letter-med'] else node_features
                    filter_n_nodes = ClassificationGraphDataset.get_filter_size(name)
                    X, A, Y = ClassificationGraphDataset.convert_tud_dataset(dset, node_features,
                                                                             filter_n_nodes=filter_n_nodes)

                    if name == 'AIDS':
                        ordered_labels = ['C', 'O', 'N', 'Cl', 'F', 'S', 'Se', 'P', 'Na', 'I', 'Co', 'Br', 'Li', 'Si',
                                          'Mg',
                                          'Cu', 'As', 'B', 'Pt', 'Ru', 'K', 'Pd', 'Au', 'Te', 'W', 'Rh', 'Zn', 'Bi',
                                          'Pb',
                                          'Ge', 'Sb', 'Sn', 'Ga', 'Hg', 'Ho', 'Tl', 'Ni', 'Tb']
                        label_map = {label: i + 1 for i, label in enumerate(ordered_labels)}
                    elif name == 'MUTAG':
                        ordered_labels = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
                        label_map = {label: i + 1 for i, label in enumerate(ordered_labels)}
                        X, A = ClassificationGraphDataset.clear_aromatic_molecule_bonds_from_dataset(X, A, label_map)
                    else:
                        label_map = None

                    if 'Letter' in name:
                        # remove 0 adj and graph with unlinked node
                        bad_graphs = []
                        for i, adj_mat in enumerate(A):
                            if adj_mat[:-1].sum() == 0:
                                bad_graphs.append(i)
                            else:
                                for j, feat in enumerate(X[i]):
                                    if feat.sum() != 0 and adj_mat[:-1, j].sum() == 0:
                                        bad_graphs.append(i)
                                        break

                        X = np.delete(X, bad_graphs, axis=0)
                        A = np.delete(A, bad_graphs, axis=0)
                        Y = np.delete(Y, bad_graphs, axis=0)
                else:
                    assert False, 'unknown dataset'

                # if add_feature is not None and add_feature > 0:
                #     # TEST ADD ZERO FEATURES IN X
                #     n_X = np.zeros((X.shape[0], X.shape[1], X.shape[-1] + add_feature))
                #     n_X[:, :, :-add_feature] = X
                #     X = n_X

                results = ((X, A, Y), val_dataset, test_dataset)

                np.save(f'{path}/{full_name}_X.npy', X)
                np.save(f'{path}/{full_name}_A.npy', A)
                np.save(f'{path}/{full_name}_Y.npy', Y)

        return results, label_map

    def is_regression_dataset(self):
        return False

    def reduce_dataset(self, reduce_type, label=None, how_many=None, reduce_from_ori=True):
        raise NotImplementedError
