from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import numpy as np
import math
import copy
import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
import scipy
import pickle

from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, make_moons, make_swiss_roll
from utils import visualize_flow

DATASETS = ['single_moon', 'double_moon', 'iris', 'bcancer', 'mnist', 'waterquality']
REGRESSION_DATASETS = ['swissroll', 'diabetes', 'waterquality']


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
            val_dataset.X = self.X[val_idx]
            val_dataset.true_labels = self.true_labels[val_idx]

            train_idx = [idx for idx in range(self.X.shape[0]) if idx not in val_idx]
            train_dataset.X = self.X[train_idx]
            train_dataset.true_labels = self.true_labels[train_idx]

            if self.idx is None:
                val_dataset.idx = val_idx
                train_dataset.idx = train_idx
            else:
                val_dataset.idx = self.idx[val_idx]
                train_dataset.idx = self.idx[train_idx]

        return train_dataset, val_dataset

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

    def __getitem__(self, idx):
        return self.get_func(idx)

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
        if exist_dataset:
            X = np.load(f'{path}/{name}_data.npy')
            labels = np.load(f'{path}/{name}_labels.npy')
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
            else:
                assert False, 'unknown dataset'

            np.save(f'{path}/{name}_data.npy', X)
            np.save(f'{path}/{name}_labels.npy', labels)

        return X, labels, test_dataset

    @staticmethod
    def format_data(input, n_bits, n_bins):
        return input.float()

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
    def format_data(input, n_bits, n_bins):
        input = input * 255

        if n_bits < 8:
            input = torch.floor(input / 2 ** (8 - n_bits))

        input = input / n_bins - 0.5
        input = input + torch.rand_like(input) / n_bins
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
