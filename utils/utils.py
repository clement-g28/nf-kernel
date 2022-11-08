import random
import torch
import os
import numpy as np
from PIL import Image


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


def save_every_pic(path, numpy_pics, methods, labels, add_str=None, clamp_min=0, clamp_max=255):
    create_folder(path)
    add_str = '' if add_str is None else f'_{add_str}'
    numpy_pics = np.clip(numpy_pics, clamp_min, clamp_max)
    for i, pic in enumerate(numpy_pics):
        im = pic.squeeze().astype(np.uint8)
        im = Image.fromarray(im, mode='L')
        name = f"{methods[i]}_{str(labels[i])}{add_str}.png"
        im.save(f'{path}/{name}')


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
