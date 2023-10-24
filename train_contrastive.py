import time

from tqdm import tqdm
import numpy as np
import math
import torch
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

from utils.graphs.utils_datasets import batch_graph_permutation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args, model_single, add_path, train_dataset, val_dataset=None):
    model = nn.DataParallel(model_single)
    model = model.to(device)

    learnable_params = list(model.parameters())
    optimizer = optim.Adam(learnable_params, lr=args.lr)

    save_dir = f"./checkpoint/{add_path}"
    create_folder(save_dir)
    create_folder(f'{save_dir}/sample')

    if args.use_tb:
        writer = SummaryWriter(save_dir)

    beta = args.beta
    # z_shape = model_single.calc_last_z_shape(dataset.im_size)

    # TEST with weighted sampler
    train_loader = train_dataset.get_loader(args.batch_size, shuffle=True, drop_last=True, sampler=True)
    if val_dataset is not None:
        val_loader = val_dataset.get_loader(args.batch_size, shuffle=True, drop_last=False)
    loader_size = len(train_loader)

    # define an instance of the PairwiseDistance
    pdist = torch.nn.PairwiseDistance(p=2)
    k = train_dataset.get_n_dim()

    # Let the distance target_D be the distance for which we have a probability of target_p
    target_D = (torch.ones(args.batch_size) * 0.005).to(device)
    target_p = 0.1
    sigma = ((k * torch.pow(target_D / 2, 2)) / math.log(target_p)).abs()
    determinant = (k * sigma)

    inv_id = torch.eye(k, k).to(device)
    inverse_matrix = torch.einsum('ij,k->kij', inv_id, 1 / sigma)

    print(f'Target log_p : {math.log(target_p):.04} in order to have a prob of {target_p} with a mean distance '
          f'of {target_D.mean().item():.04} using a sigma of {sigma.mean().item():.04}.')

    with tqdm(range(args.n_epoch)) as pbar:
        for epoch in pbar:
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                model_single.downstream_process()
                itr = epoch * loader_size + i
                input, label = data

                input = train_dataset.format_data(input, device)
                input_p = batch_graph_permutation(input)
                label = label.to(device)

                if itr == 0:
                    with torch.no_grad():
                        log_p, distloss, logdet, o = model.module(input, label)

                        continue

                else:
                    # nll1 + dist
                    log_p, distloss, logdet, o = model(input, label)

                    # nll2
                    log_p2, _, logdet2, o2 = model(input_p, label)
                    # nllmean
                    o_mean = torch.mean(torch.cat((o.unsqueeze(0), o2.unsqueeze(0)), 0), 0)

                    # compute the pairwise distance
                    D = pdist(o, o2).mean()

                    # D = torch.diag(torch.cdist(o, o2))

                    b_size = o_mean.shape[0]
                    k = o_mean.shape[1]

                    # p*max computing approx (0.9,0.01) (0.1,10) (p*max, D)
                    # p_max = torch.nan_to_num(torch.clamp(0.85 * torch.log10(-D + 11.5), min=math.pow(10, -10)), 0.1)
                    # p_max = torch.nan_to_num(torch.clamp(0.75 * torch.log10(-D + 21.5), min=0.1), 0.1)
                    # sigma computing
                    # sigma = torch.clamp(torch.sqrt(-torch.pow(D / 2, 2) / (2 * torch.log(p_max))), max=5)
                    # sigma = torch.ones_like(sigma) * 0.001

                    # inverse_matrix = torch.einsum('ij,k->kij', inv_id, 1 / sigma)

                    log_p_o_mean = -0.5 * (k * math.log(2 * math.pi) + torch.diag(
                        (torch.diagonal((o - o_mean) @ inverse_matrix, offset=0, dim1=0, dim2=1).transpose(1,
                                                                                                           0).unsqueeze(
                            0) @ torch.transpose(o - o_mean, 1, 0)).reshape(b_size, -1), 0)) - torch.log(determinant)
                    log_p_o_mean = log_p_o_mean.unsqueeze(1)
                    log_p_o2_mean = -0.5 * (k * math.log(2 * math.pi) + torch.diag(
                        (torch.diagonal((o2 - o_mean) @ inverse_matrix,
                                        offset=0, dim1=0, dim2=1).transpose(1, 0).unsqueeze(0) @ torch.transpose(
                            o2 - o_mean, 1, 0)).reshape(b_size, -1), 0)) - torch.log(determinant)
                    log_p_o2_mean = log_p_o2_mean.unsqueeze(1)

                    # log_p_loss = torch.mean(torch.cat((log_p.unsqueeze(0), log_p_o_mean.unsqueeze(0)), 0), 0)
                    # log_p2_loss = torch.mean(torch.cat((log_p2.unsqueeze(0), log_p_o2_mean.unsqueeze(0)), 0), 0)

                # logdet = logdet.mean()

                # nll_loss, log_p, log_det = calc_loss(log_p, logdet, dataset.im_size, n_bins)
                nll_o1_loss, log_p, log_det = train_dataset.format_loss(log_p, logdet)
                nll_o1_mean_loss, log_p_o_mean, log_det = train_dataset.format_loss(log_p_o_mean, logdet)
                nll_o2_loss, log_p2, log_det2 = train_dataset.format_loss(log_p2, logdet2)
                nll_o2_mean_loss, log_p_o2_mean, log_det2 = train_dataset.format_loss(log_p_o2_mean, logdet2)

                loss_o1 = nll_o1_loss + nll_o1_mean_loss
                loss_o2 = nll_o2_loss + nll_o2_mean_loss

                # loss = loss_o1 + loss_o2 - beta * distloss
                loss = nll_o1_mean_loss + nll_o2_mean_loss

                loss = model_single.upstream_process(loss)
                # loss.clamp_(-10000,10000)

                model.zero_grad()
                # loss.backward()

                # if epoch < 100:
                #     loss += log_det
                # loss.backward(retain_graph=True)
                # loss2 = torch.pow(log_det, 2)
                # loss2.backward()

                # if 10 < epoch < 50:
                #     loss.backward(retain_graph=True)
                #     loss_tmp = torch.pow(log_det - 500,2) + torch.pow(log_det2 - 500,2)
                #     loss_tmp.backward()
                # else:
                loss.backward()
                # Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
                warmup_lr = args.lr
                optimizer.param_groups[0]["lr"] = warmup_lr
                optimizer.step()

                pbar.set_description(
                    f"Loss: {loss.item():.4f}; "
                    f"logP1: {log_p.mean().item():.4f}; logdet1: {log_det.item():.4f}; "
                    f"logP2: {log_p2.mean().item():.4f}; logdet2: {log_det2.item():.4f}; "
                    f"D: {D.mean().item():.4f}; logP1_con: {log_p_o_mean.mean().item():.4f}; logP2_con: {log_p_o2_mean.mean().item():.4f}; "
                    f"lr: {warmup_lr:.5f}; Lmu: {distloss.item():.5f}"
                )

                if itr % args.write_train_loss_every == 0:
                    if args.use_tb:
                        loss_dict = {"tot_loss": loss.item(),
                                     "logP1": log_p.mean().item(),
                                     "logP2": log_p2.mean().item(),
                                     "logdet": log_det.item(),
                                     "logdet2": log_det.item(),
                                     "logP1_Mean": log_p_o_mean.mean().item(),
                                     "logP2_Mean": log_p_o2_mean.mean().item(),
                                     "D": D.mean().item(),
                                     # "P_max": p_max.mean().item(),
                                     # "Sigma": sigma.mean().item(),
                                     "distloss": distloss.item()}
                        write_dict_to_tensorboard(writer, loss_dict, base_name="TRAIN", iteration=itr)

                        means_dict = {}
                        for i, mean in enumerate(model_single.means):
                            m = np.mean(mean.detach().cpu().numpy())
                            means_dict[f'mean{i}'] = m.item()
                        write_dict_to_tensorboard(writer, means_dict, base_name=f"MEANS", iteration=itr)

                if itr % args.sample_every == 0:
                    with torch.no_grad():
                        model.eval()
                        model_single.sample_evaluation(itr, train_dataset, val_dataset, save_dir,
                                                       writer=writer)
                        model.train()

                del loss
                torch.cuda.empty_cache()

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

                        input = val_dataset.format_data(input, device)
                        label = label.to(device)

                        log_p, distloss, logdet, z = model(input, label)

                        logdet = logdet.mean()

                        # nll_loss, log_p, log_det = calc_loss(log_p, logdet, dataset.im_size, n_bins)
                        nll_loss, log_p, log_det = val_dataset.format_loss(log_p, logdet)
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

            if epoch > args.save_at_epoch and epoch % args.save_each_epoch == 0:
                torch.save(model.state_dict(), f"{save_dir}/model_{str(itr + 1).zfill(6)}.pt")


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
    dataset = load_dataset(args, args.dataset, args.model, transform=transform, add_feature=args.add_feature)

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

    # reduce train dataset size (fitting too long)
    print('Train dataset reduced in order to accelerate. (stratified)')
    train_dataset.reduce_dataset_ratio(0.05, stratified=True)

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
            # step = border * 2 / dim_per_label
            x = np.linspace(-border, border, dim_per_label)
            # if (dim_per_label % 2) != 0 else np.concatenate(
            # (np.linspace(-border, g_param[0], int(dim_per_label / 2))[:-1], [g_param[0]],
            #  np.linspace(step, border, int(dim_per_label / 2))))
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
        gaussian_params = initialize_class_gaussian_params(dataset, eigval_list, isotrope=args.isotrope_gaussian,
                                                           dim_per_label=dim_per_label, fixed_eigval=fixed_eigval,
                                                           split_graph_dim=args.split_graph_dim)
    else:
        if args.method == 0:
            gaussian_params = initialize_regression_gaussian_params(dataset, eigval_list,
                                                                    isotrope=args.isotrope_gaussian,
                                                                    dim_per_label=dim_per_label,
                                                                    fixed_eigval=fixed_eigval)
        elif args.method == 1:
            gaussian_params = initialize_tmp_regression_gaussian_params(dataset, eigval_list)
        elif args.method == 2:
            gaussian_params = initialize_tmp_regression_gaussian_params(dataset, eigval_list, ones=True)
        else:
            assert False, 'no method selected'

    reg_use_var_str = f''
    if args.reg_use_var:
        reg_use_var_str = f'_usevar'

    split_graph_dim_str = f''
    if args.split_graph_dim and isinstance(dataset, GraphDataset) and not dataset.is_regression_dataset():
        split_graph_dim_str = f'_splitgraphdim'

    folder_path = f'{args.dataset}/{args.model}/'
    if args.model == 'cglow':
        args_cglow, _ = cglow_arguments().parse_known_args()
        n_channel = dataset.n_channel
        model_single = load_cglow_model(args_cglow, n_channel, gaussian_params=gaussian_params,
                                        learn_mean=not args.fix_mean, device=device)
        folder_path += f'b{args_cglow.n_block}_f{args_cglow.n_flow}'
    elif args.model == 'seqflow':
        args_seqflow, _ = seqflow_arguments().parse_known_args()
        model_single = load_seqflow_model(dataset.in_size, args_seqflow.n_flow, gaussian_params=gaussian_params,
                                          learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
        folder_path += f'f{args_seqflow.n_flow}'
    elif args.model == 'ffjord':
        args_ffjord, _ = ffjord_arguments().parse_known_args()
        # args_ffjord.n_block = args.n_block
        model_single = load_ffjord_model(args_ffjord, dataset.in_size, gaussian_params=gaussian_params,
                                         learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
        folder_path += f'b{args_ffjord.n_block}'
    elif args.model == 'moflow':
        args_moflow, _ = moflow_arguments().parse_known_args()
        model_single = load_moflow_model(args_moflow, gaussian_params=gaussian_params,
                                         learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
    elif args.model == 'graphnvp':
        args_graphnvp, _ = graphnvp_arguments().parse_known_args()
        model_single = load_graphnvp_model(args_graphnvp, gaussian_params=gaussian_params,
                                           learn_mean=not args.fix_mean, reg_use_var=args.reg_use_var, dataset=dataset)
    else:
        assert False, 'unknown model type'

    # model_single.init_means_to_points(train_dataset)

    lmean_str = f'_lmean{args.beta}' if not args.fix_mean else ''
    isotrope_str = '_isotrope' if args.isotrope_gaussian else ''
    folder_path += f'_nfkernel{lmean_str}{isotrope_str}{eigval_str}{noise_str}' \
                   f'{redclass_str}{redlabel_str}_dimperlab{dim_per_label}{reg_use_var_str}{split_graph_dim_str}'

    if args.add_in_name_folder is not None:
        folder_path += f'_{args.add_in_name_folder}'

    create_folder(f'./checkpoint/{folder_path}')

    path = f'./checkpoint/{folder_path}/train_idx'
    train_dataset.save_split(path)
    if val_dataset is not None:
        path = f'./checkpoint/{folder_path}/val_idx'
        val_dataset.save_split(path)

    restart = args.restart
    if restart:
        from utils.custom_glow import WrappedModel

        flow_path = f'./checkpoint/{folder_path}/restart.pth'
        model_single = WrappedModel(model_single)
        model_single.load_state_dict(torch.load(flow_path, map_location=device))
        model_single = model_single.module
        folder_path += '/restart'
        create_folder(f'./checkpoint/{folder_path}')

    train_start = time.time()
    train(args, model_single, folder_path, train_dataset=train_dataset, val_dataset=val_dataset)
    train_end = time.time()
    del dataset
    gpu_info = torch.cuda.mem_get_info(device=device)
    model_single.del_model_from_gpu()
    return f'./checkpoint/{folder_path}', train_end - train_start, gpu_info


if __name__ == "__main__":
    args, _ = training_arguments().parse_known_args()
    print(args)

    main(args)
