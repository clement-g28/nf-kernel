from tqdm import tqdm
import numpy as np
import math
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils

from utils.custom_glow import CGlow
from utils.toy_models import load_seqflow_model, load_ffjord_model
from utils.training import training_arguments, ffjord_arguments, AddGaussianNoise, calc_loss
from utils.dataset import ImDataset, SimpleDataset
from utils.density import construct_covariance
from utils.utils import write_dict_to_tensorboard, set_seed, create_folder, AverageMeter, initialize_gaussian_params, \
    initialize_regression_gaussian_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, model_single, add_path, gaussian_params, train_dataset, val_dataset=None):
    model = nn.DataParallel(model_single)
    model = model.to(device)

    learnable_params = list(model.parameters())
    optimizer = optim.Adam(learnable_params, lr=args.lr)

    save_dir = f"./checkpoint/{add_path}"
    create_folder(save_dir)
    create_folder(f'{save_dir}/sample')

    if args.use_tb:
        writer = SummaryWriter(save_dir)

    n_bins = 2.0 ** args.n_bits

    beta = args.beta
    z_shape = model_single.calc_last_z_shape(dataset.im_size)

    # TEST with weighted sampler
    train_loader = train_dataset.get_loader(args.batch_size, shuffle=True, drop_last=False, sampler=True)
    if val_dataset is not None:
        val_loader = val_dataset.get_loader(args.batch_size, shuffle=True, drop_last=False)
    loader_size = len(train_loader)

    with tqdm(range(args.n_epoch)) as pbar:
        for epoch in pbar:
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                model_single.downstream_process()
                itr = epoch * loader_size + i
                input, label = data
                input = input.to(device)
                label = label.to(device)

                input = dataset.format_data(input, args.n_bits, n_bins)

                if itr == 0:
                    with torch.no_grad():
                        log_p, distloss, logdet, o = model.module(input, label)

                        continue

                else:
                    log_p, distloss, logdet, o = model(input, label)

                # logdet = logdet.mean()

                # nll_loss, log_p, log_det = calc_loss(log_p, logdet, dataset.im_size, n_bins)
                nll_loss, log_p, log_det = dataset.format_loss(log_p, logdet, n_bins)
                loss = nll_loss - beta * distloss

                loss = model_single.upstream_process(loss)

                model.zero_grad()
                loss.backward()
                # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
                warmup_lr = args.lr
                optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()

                pbar.set_description(
                    f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}; distloss: {distloss.item():.7f}"
                )

                if itr % 100 == 0:
                    if args.use_tb:
                        loss_dict = {"tot_loss": loss.item(),
                                     "logP": log_p.item(),
                                     "logdet": log_det.item(),
                                     "distloss": distloss.item()}
                        write_dict_to_tensorboard(writer, loss_dict, base_name="TRAIN", iteration=itr)

                        means_dict = {}
                        for i, mean in enumerate(model_single.means):
                            m = np.mean(mean.detach().cpu().numpy())
                            means_dict[f'mean{i}'] = m.item()
                        write_dict_to_tensorboard(writer, means_dict, base_name=f"MEANS", iteration=itr)

                    with torch.no_grad():
                        model.eval()
                        model_single.sample_evaluation(itr, train_dataset, val_dataset, z_shape, save_dir,
                                                       writer=writer)
                        model.train()

            # Evaluation
            if val_dataset is not None:
                model.eval()

                valLossMeter = AverageMeter()
                valLogPMeter = AverageMeter()
                valLogDetMeter = AverageMeter()
                valDistLossMeter = AverageMeter()

                with torch.no_grad():
                    for data in val_loader:
                        input, label = data
                        input = input.to(device)
                        label = label.to(device)

                        input = dataset.format_data(input, args.n_bits, n_bins)

                        log_p, distloss, logdet, z = model(input, label)

                        logdet = logdet.mean()

                        # nll_loss, log_p, log_det = calc_loss(log_p, logdet, dataset.im_size, n_bins)
                        nll_loss, log_p, log_det = dataset.format_loss(log_p, logdet, n_bins)
                        loss = nll_loss - beta * distloss

                        valLossMeter.update(loss.item())
                        valLogPMeter.update(log_p.item())
                        valLogDetMeter.update(log_det.item())
                        valDistLossMeter.update(distloss.item())

                    if args.use_tb:
                        loss_dict = {"tot_loss": valLossMeter.avg,
                                     "logP": valLogPMeter.avg,
                                     "logdet": valLogDetMeter.avg,
                                     "distloss": distloss.item()}
                        write_dict_to_tensorboard(writer, loss_dict, base_name="VAL", iteration=itr)

                    with torch.no_grad():
                        model_single.evaluate(itr, input, label, z, val_dataset, save_dir, writer=writer)
                        # images = model_single.reverse(z).cpu().data
                        # images = train_dataset.rescale(images)
                        # utils.save_image(
                        #     images,
                        #     f"{save_dir}/sample/val_{str(itr + 1).zfill(6)}.png",
                        #     normalize=True,
                        #     nrow=10,
                        #     range=(0, 255),
                        # )
                        # img_grid = utils.make_grid(images, nrow=10, padding=2, pad_value=0, normalize=True,
                        #                            range=(0, 255), scale_each=False)
                        # writer.add_image('val', img_grid)
                model.train()

            if epoch % args.save_each_epoch == 0:
                torch.save(model.state_dict(), f"{save_dir}/model_{str(itr + 1).zfill(6)}.pt")


if __name__ == "__main__":
    args, _ = training_arguments().parse_known_args()
    print(args)

    set_seed(0)

    if args.dataset == 'mnist':
        transform = [
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    elif args.dataset == 'cifar10':
        transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    else:
        transform = []

    noise_str = ''
    if args.with_noise is not None:  # ((dataset.X / 255 -dataset.norm_mean) / dataset.norm_std).std() *1/5 =.2
        transform += [AddGaussianNoise(0., args.with_noise)]
        noise_str = '_noise' + str(args.with_noise).replace('.', '')

    transform = transforms.Compose(transform)

    if args.dataset == 'mnist':
        dataset = ImDataset(dataset_name=args.dataset, transform=transform)
    else:
        dataset = SimpleDataset(dataset_name=args.dataset, transform=transform)

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

    validation = args.validation
    if validation > 0:
        # TEST with stratified sample
        train_dataset, val_dataset = dataset.split_dataset(validation, stratified=True)
        train_dataset.ori_X = train_dataset.X
        train_dataset.ori_true_labels = train_dataset.true_labels
        val_dataset.ori_X = val_dataset.X
        val_dataset.ori_true_labels = val_dataset.true_labels
    else:
        train_dataset = dataset
        val_dataset = None

    device = 'cuda:0'

    n_dim = dataset.X[0].shape[0]
    for sh in dataset.X[0].shape[1:]:
        n_dim *= sh

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
    eigval_str = ''
    if args.set_eigval_manually is None:
        mean_of_eigval_str = str(args.mean_of_eigval).replace('.', '-')
        if args.uniform_eigval:
            eigval_list = [args.mean_of_eigval for i in range(dim_per_label)]
            eigval_str = f'_eigvaluniform{mean_of_eigval_str}'
        elif args.gaussian_eigval is not None:
            g_param = list(map(float, args.gaussian_eigval.strip('[]').split(',')))
            assert len(g_param) == 2, 'gaussian_eigval should be composed of 2 float the mean and the std'
            import scipy.stats as st

            dist = st.norm(loc=g_param[0], scale=g_param[1])
            border = 1.6
            step = border * 2 / dim_per_label
            x = np.linspace(-border, border, dim_per_label) if (dim_per_label % 2) != 0 else np.concatenate(
                (np.linspace(-border, g_param[0], int(dim_per_label / 2))[:-1], [g_param[0]],
                 np.linspace(step, border, int(dim_per_label / 2))))
            eigval_list = dist.pdf(x)
            mean_eigval = args.mean_of_eigval
            a = mean_eigval * dim_per_label / eigval_list.sum()
            eigval_list = a * eigval_list
            eigval_list[np.where(eigval_list < 1)] = 1
            std_str = str(g_param[1]).replace('.', '-')
            eigval_str = f'_eigvalgaussian{mean_of_eigval_str}std{std_str}'
        else:
            assert False, 'No distribution selected; use uniform_eigval or gaussian_eigval arguments'
    else:
        fixed_eigval = list(map(float, args.set_eigval_manually.strip('[]').split(',')))
        eigval_list = None
        eigval_str = args.set_eigval_manually.strip('[]').replace(',', '-')
        eigval_str = f'_manualeigval{eigval_str}'

    if not dataset.is_regression_dataset():
        gaussian_params = initialize_gaussian_params(dataset, eigval_list, isotrope=args.isotrope_gaussian,
                                                     dim_per_label=dim_per_label, fixed_eigval=fixed_eigval)
    else:
        gaussian_params = initialize_regression_gaussian_params(dataset, eigval_list, isotrope=args.isotrope_gaussian,
                                                                dim_per_label=dim_per_label, fixed_eigval=fixed_eigval)
    folder_path = f'{args.dataset}/{args.model}/'
    if args.model == 'cglow':
        n_channel = dataset.n_channel
        model_single = CGlow(n_channel, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu,
                             gaussian_params=gaussian_params, device=device, learn_mean=not args.fix_mean)
        folder_path += f'b{args.n_block}_f{args.n_flow}'
    elif args.model == 'seqflow':
        model_single = load_seqflow_model(dataset.im_size, args.n_flow, gaussian_params=gaussian_params,
                                          learn_mean=not args.fix_mean, dataset=dataset)
        folder_path += f'f{args.n_flow}'
    elif args.model == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        args_ffjord.n_block = args.n_block
        model_single = load_ffjord_model(args_ffjord, dataset.im_size, gaussian_params=gaussian_params,
                                         learn_mean=not args.fix_mean, dataset=dataset)
        folder_path += f'b{args.n_block}'
    else:
        assert False, 'unknown model type'

    lmean_str = f'_lmean{args.beta}' if not args.fix_mean else ''
    isotrope_str = '_isotrope' if args.isotrope_gaussian else ''
    folder_path += f'_nfkernel{lmean_str}{isotrope_str}{eigval_str}{noise_str}' \
                   f'{redclass_str}{redlabel_str}_dimperlab{dim_per_label}'

    create_folder(f'./checkpoint/{folder_path}')

    path = f'./checkpoint/{folder_path}/train_idx'
    train_dataset.save_split(path)
    if val_dataset is not None:
        path = f'./checkpoint/{folder_path}/val_idx'
        val_dataset.save_split(path)

    train(args, model_single, folder_path, gaussian_params=gaussian_params, train_dataset=train_dataset,
          val_dataset=val_dataset)
