import datetime
import os
import time

import pandas as pd
import torch
import torch.utils.data
from torch import nn
import torchvision
from torch.utils.data import random_split
from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter
import math
from torch.cuda import amp
import torch.distributed.optim
import argparse
from torchvision import datasets, transforms
from spikingjelly.clock_driven import functional
import utils
from tqdm import tqdm
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torch.optim.lr_scheduler import CosineAnnealingLR
_seed_ = 2020
import random
import torch.optim as optim
from tensorboardX import SummaryWriter
from spikingjelly.clock_driven.monitor import Monitor
from snnvgg import *
#from spiking_resnet_p import *
#from snnwrn import *
#from snnwrn_p import *
from torch.autograd import Variable
print('4')
writer = SummaryWriter('./')
random.seed(2020)



from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np

writer = SummaryWriter('./')


from mask_manage import PruningLayer, PruningNetworkManager
np.random.seed(_seed_)

import numpy as np
from snndvs import SNNDVS5Conv

np.random.seed(_seed_)


import copy
import torch.nn.functional as F
from spikingjelly.clock_driven import functional as sj_func





def load_teacher_model(teacher_path, device, num_classes, args):
    """
    从 checkpoint 加载 teacher 模型
    """
    if not os.path.isfile(teacher_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    # 按你当前训练用的模型结构创建 teacher
    if args.dataset == 'dvscifar10':
        teacher = SNNDVS5Conv(num_classes=10, in_channels=2)
    else:
        teacher = snnvgg16_bn(num_classes=num_classes)

    checkpoint = torch.load(teacher_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    teacher.load_state_dict(state_dict, strict=True)
    teacher = teacher.to(device)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad_(False)

    return teacher















def _make_calib_iter(data_loader, num_batches):
    """只取前 num_batches 个 batch 用来校准/重建，避免太耗时"""
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        yield batch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_connection_percent(model):
    total_weights = 0
    nonzero_weights = 0

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.data
            total_weights += w.numel()
            nonzero_weights += (w != 0).sum().item()

    percent = 100.0 * nonzero_weights / total_weights
    return nonzero_weights, total_weights, percent



def load_cfg_channels(cfg_path):
    with open(cfg_path, "r") as f:
        text = f.read().strip()
    # cfg.txt 形如: [52, 60, 110, ...]
    cfg_channels = eval(text)
    return cfg_channels







@torch.no_grad()


def compute_vgg_synops(cfg_channels, spike_rates, T=4, input_size=32, num_classes=100):
    """
    按压缩后模型计算 SynOps
    只统计:
    - 13个卷积层
    - 3个全连接层(classifier1, classifier2, fc)

    关键改动:
    每一层卷积的 SynOps 使用“输入到该层的 spike rate”
    而不是当前层输出 spike rate
    """
    assert len(cfg_channels) == 13, f"Expected 13 conv layers, got {len(cfg_channels)}"
    assert len(spike_rates) == 13, f"Expected 13 spike rates, got {len(spike_rates)}"

    if input_size == 32:
        spatial_sizes = [32, 32, 16, 16, 8, 8, 8, 4, 4, 4, 2, 2, 2]
    elif input_size == 64:
        spatial_sizes = [64, 64, 32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4]
    else:
        raise ValueError(f"Unsupported input_size: {input_size}")

    total_synops = 0.0
    in_channels = 3

    # conv1 输入是图像，输入率设为 1.0
    # conv2 用第1层输出spike rate
    # conv3 用第2层输出spike rate
    # ...
    input_spike_rates = [1.0] + list(spike_rates[:-1])

    # 13个卷积层
    for i, out_channels in enumerate(cfg_channels):
        H = W = spatial_sizes[i]
        r_in = input_spike_rates[i]
        synops = T * H * W * out_channels * in_channels * 3 * 3 * r_in
        total_synops += synops
        in_channels = out_channels

    # FC部分
    last_c = cfg_channels[-1]

    # classifier1: last_c -> 512
    r_fc1_in = spike_rates[-1]
    total_synops += T * last_c * 512 * r_fc1_in

    # classifier2: 512 -> 512
    # 这里先近似使用最后一层卷积输出的 spike rate
    r_fc2_in = spike_rates[-1]
    total_synops += T * 512 * 512 * r_fc2_in

    # fc: 512 -> num_classes
    r_fc3_in = spike_rates[-1]
    total_synops += T * 512 * num_classes * r_fc3_in

    return total_synops








def compute_compact_vgg_params(cfg_channels, num_classes=100):
    """
    计算压缩后 VGG16 SNN 的参数量
    只统计:
    - 所有 Conv 层
    - 所有 FC 层
    不统计 BN
    与论文 Params = sum(Params_conv) + sum(Params_fc) 一致
    """

    assert len(cfg_channels) == 13, f"Expected 13 conv channels, got {len(cfg_channels)}"

    total_params = 0
    in_channels = 3

    # 13个卷积层参数
    for out_channels in cfg_channels:
        total_params += out_channels * in_channels * 3 * 3
        in_channels = out_channels

    # 分类头参数
    last_c = cfg_channels[-1]

    # classifier1: Linear(last_c -> 512, bias=False)
    total_params += last_c * 512

    # classifier2: Linear(512 -> 512, bias=False)
    total_params += 512 * 512

    # fc: Linear(512 -> num_classes, bias=True)
    total_params += 512 * num_classes + num_classes

    return total_params


'''
from thop import profile

def compute_macs(model, input_shape=(1,3,32,32), device='cuda'):
    model.eval()
    dummy = torch.randn(input_shape).to(device)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    return macs
'''



def _gap_time(feat: torch.Tensor):
    """
    SNN feature: [T, B, C, H, W] -> [B, C]
    ANN feature: [B, C, H, W] -> [B, C]
    """
    if feat.dim() == 5:
        feat = feat.mean(dim=0)       # [B, C, H, W]
    if feat.dim() == 4:
        feat = feat.mean(dim=(2, 3))  # GAP -> [B, C]
    return feat




def _register_feature_hooks(model, layer_names):
    """
    layer_names: 例如 ('layer12', 'layer13')
    返回:
        features: dict
        handles: list
    """
    features = {}
    handles = []

    name_to_module = dict(model.named_modules())

    for name in layer_names:
        if name not in name_to_module:
            raise ValueError(f'Hook layer "{name}" not found in model.named_modules().')

        def _make_hook(key):
            def hook(module, inp, out):
                features[key] = out
            return hook

        h = name_to_module[name].register_forward_hook(_make_hook(name))
        handles.append(h)

    return features, handles




def _normalize_list(vals):
    vals = torch.tensor(vals, dtype=torch.float32)
    if vals.numel() == 0:
        return vals
    vmin = vals.min()
    vmax = vals.max()
    if float(vmax - vmin) < 1e-12:
        return torch.zeros_like(vals)
    return (vals - vmin) / (vmax - vmin + 1e-6)


def compute_adaptive_layer_weights(manager):
    """
    增强版:
    w_l = (0.7 * D_hat + 0.3 * S_hat) * (1 + 0.5 * H_hat)
    """
    D_list = manager.get_avg_change_ratios()
    S_list = manager.get_avg_spike_rates()
    H_list = manager.get_depth_priors()

    D_hat = _normalize_list(D_list)
    S_hat = _normalize_list(S_list)
    H_hat = _normalize_list(H_list)

    weights = (0.7 * D_hat + 0.3 * S_hat) * (1.0 + 0.5 * H_hat)
    return weights.tolist()


def get_last_conv_names(model, num_layers=4):
    conv_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_names.append(name)
    if len(conv_names) <= num_layers:
        return tuple(conv_names)
    return tuple(conv_names[-num_layers:])








def adaptive_final_reconstruction(model, teacher, manager, calib_loader, device,
                                  criterion, iters=20, lr=1e-3,
                                  lambda_feat=1.0, topk=3, use_amp=False):
    """
    训练全部结束后执行一次自适应特征重建：
    1. 根据动态扰动 + spike rate + 深层先验算权重
    2. 从最后若干个卷积层中选 top-k 做重建
    3. 每步优化后重新施加 mask，防止已剪通道复活
    """
    model.train()
    teacher.eval()

    # 冻结 BN running stats，避免小校准集带偏统计
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    candidate_layers = get_last_conv_names(model, num_layers=4)
    print('Adaptive REC candidate layers:', candidate_layers)

    # 先算所有层权重
    all_layer_weights = compute_adaptive_layer_weights(manager)

    # 构造 conv name -> idx 的映射，顺序要和 do_masks 的 Conv 顺序一致
    conv_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_names.append(name)
    conv_name_to_idx = {name: idx for idx, name in enumerate(conv_names)}

    candidate_weight_dict = {}
    for name in candidate_layers:
        idx = conv_name_to_idx[name]
        candidate_weight_dict[name] = all_layer_weights[idx]

    # 选 top-k 层
    sorted_items = sorted(candidate_weight_dict.items(), key=lambda x: x[1], reverse=True)
    selected_items = sorted_items[:min(topk, len(sorted_items))]
    selected_layers = tuple([x[0] for x in selected_items])
    selected_weight_dict = {k: v for k, v in selected_items}

    print('Adaptive REC selected layers:', selected_layers)
    print('Adaptive REC weights:', selected_weight_dict)

    # 注册 hook
    s_feats, s_handles = _register_feature_hooks(model, selected_layers)
    t_feats, t_handles = _register_feature_hooks(teacher, selected_layers)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    for x, y in calib_loader:
        x = x.to(device).float()
        y = y.to(device)

        with torch.no_grad():
            _ = teacher(x)
            sj_func.reset_net(teacher)
            t_feature_dict = {}
            for name in selected_layers:
                t_feature_dict[name] = _gap_time(t_feats[name]).detach()
            sj_func.reset_net(teacher)

        if use_amp:
            with amp.autocast():
                s_logits = model(x)
                s_feature_dict = {}
                for name in selected_layers:
                    s_feature_dict[name] = _gap_time(s_feats[name])
                sj_func.reset_net(model)

                loss_feat = 0.0
                for name in selected_layers:
                    w = selected_weight_dict[name]
                    loss_feat = loss_feat + w * F.mse_loss(s_feature_dict[name], t_feature_dict[name])

                loss_ce = criterion(s_logits, y)
                loss = lambda_feat * loss_feat + 0.1 * loss_ce
        else:
            s_logits = model(x)
            s_feature_dict = {}
            for name in selected_layers:
                s_feature_dict[name] = _gap_time(s_feats[name])
            sj_func.reset_net(model)

            loss_feat = 0.0
            for name in selected_layers:
                w = selected_weight_dict[name]
                loss_feat = loss_feat + w * F.mse_loss(s_feature_dict[name], t_feature_dict[name])

            loss_ce = criterion(s_logits, y)
            loss = lambda_feat * loss_feat + 0.1 * loss_ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # 关键：防止已剪通道复活
        manager.do_masks(model)

        step += 1
        if step >= iters:
            break

    for h in s_handles + t_handles:
        h.remove()




import csv
l1_lambda=3e-5

def l1_regularization(model, l1_alpha):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
        
            module.weight.grad.data.add_(l1_alpha * torch.sign(module.weight.data))


def train_one_epoch(model, manager, criterion, optimizer, data_loader, device, epoch, print_freq,
                    scaler=None, accum_steps=1):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    accum_steps = max(1, accum_steps)
    optimizer.zero_grad(set_to_none=True)

    for step, (image, target) in enumerate(tqdm(data_loader), start=1):

        start_time = time.time()
        image, target = image.to(device).float(), target.to(device)

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)

        # 关键：缩放 loss
        loss_to_backward = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        # 每 accum_steps 次更新一次
        if step % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                l1_regularization(model, l1_lambda)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        functional.reset_net(model)

        # ===== logging =====
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]

        loss_s = loss.item()
        if math.isnan(loss_s):
            raise ValueError('loss is Nan')

        metric_logger.update(loss=loss_s, lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))

    # 处理最后剩余梯度
    if step % accum_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            l1_regularization(model, l1_lambda)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    metric_logger.synchronize_between_processes()

    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg




def evaluate(model, criterion, data_loader, device, print_freq=100, header='Test:'):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    correct = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            
            functional.reset_net(model)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    

    loss, acc1, acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f' * Acc@1 = {acc1}, Acc@5 = {acc5}, loss = {loss}')
    
    return loss, acc1, acc5


def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path















def main(args):
    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.

    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    print(args)
    output_dir = os.path.join(args.output_dir, f'{args.model}_b{args.batch_size}_lr{args.lr}_T{args.T}')

    if args.zero_init_residual:
        output_dir += '_zi'
    if args.weight_decay:
        output_dir += f'_wd{args.weight_decay}'

    output_dir += f'_coslr{args.cos_lr_T}'

    if args.adam:
        output_dir += '_adam'
    else:
        output_dir += '_sgd'

    if args.connect_f:
        output_dir += f'_cnf_{args.connect_f}'

    if output_dir:
        utils.mkdir(output_dir)

    device = torch.device(args.device)

   


    '''
    tr = datasets.CIFAR10('./', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomCrop(32, padding=4),
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                          ]))

    te = datasets.CIFAR10('./', train=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                          ]))
    
    '''
    'tiny_imagenet'

    is_dvs = False

    if args.dataset == 'cifar10':
        num_classes = 10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610)
        Dataset = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        num_classes = 100
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        Dataset = datasets.CIFAR100
    elif args.dataset == 'dvscifar10':
        num_classes = 10

        is_dvs = True
    elif args.dataset == 'tiny_imagenet':
        num_classes = 200
        is_dvs = False  # 显式写清楚
        # Tiny-ImageNet 通常用 ImageNet mean/std
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')







    if not is_dvs:

        if args.dataset in ['cifar10', 'cifar100']:
            train_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

            tr = Dataset('./', train=True, download=True, transform=train_tf)
            te = Dataset('./', train=False, download=True, transform=test_tf)

        elif args.dataset == 'tiny_imagenet':
            train_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            test_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

            train_dir = os.path.join(args.data_path, 'train')
            val_dir = os.path.join(args.data_path, 'val')

            tr = datasets.ImageFolder(train_dir, transform=train_tf)
            te = datasets.ImageFolder(val_dir, transform=test_tf)

        else:
            raise ValueError(f'Unexpected non-DVS dataset: {args.dataset}')

        '''
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


        tr = Dataset('./', train=True, download=True, transform=train_tf)
        te = Dataset('./', train=False, download=True, transform=test_tf)
        '''

    else:

    # DVS-CIFAR10: 使用 SpikingJelly 的 frame 数据

        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
        full = CIFAR10DVS(
            root='./download/cifar10dvs',
            data_type='frame',
            frames_number=args.frames_number,
            split_by='number'
        )

        n = len(full)
        n_train = int(0.9 * n)
        n_test = n - n_train

        tr, te = random_split(full, [n_train, n_test])



    print(f'dataset_train:{tr.__len__()}, dataset_test:{te.__len__()}')
    train_loader = torch.utils.data.DataLoader(
        tr,
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    '''
    test_loader = torch.utils.data.DataLoader(
        te,
        batch_size=args.batch_size, shuffle=False, drop_last=True)
    '''
    test_loader = torch.utils.data.DataLoader(
        te,
        batch_size=args.batch_size, shuffle=False, drop_last=True)

    print("Creating model")

    if args.dataset == 'dvscifar10':
        model = SNNDVS5Conv(num_classes=10, in_channels=2).to(device)
    else:
        model=snnvgg16_bn(num_classes=num_classes).to(device)
    mymanager = PruningNetworkManager(model, args.output_dir)

    orig_params = count_parameters(model)
    print(f"Original Params: {orig_params}")

    
    print('model')

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    print('abcdef12346')
    criterion = nn.CrossEntropyLoss()
    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    print('wwww')
    

    if args.amp:
        scaler = amp.GradScaler()
    else:
        print('n')
        scaler = None

    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cos_lr_T)
    '''
    lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    '''
    if args.scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        '''
        lr_scheduler = MultiStepLR(
            optimizer,
            milestones=[100, 150],
            gamma=0.1
        )
        '''
        lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.resume:
        print('a')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

        max_test_acc1 = checkpoint['max_test_acc1']
        evaluate(model, criterion, test_loader, device=device, header='Test:')
        return

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    print("Start training")
    start_time = time.time()

    
    for epoch in range(args.epochs):
        print(epoch)
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mymanager.training()
        train_loss, train_acc1, train_acc5 = train_one_epoch(model, mymanager,criterion, optimizer, train_loader, device, epoch,
                                                             args.print_freq, scaler,accum_steps=args.accum_steps)
        '''if utils.is_main_process():
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)'''
        lr_scheduler.step()
        print(train_acc1)
        
        
        mymanager.evaling()

        test_loss, test_acc1, test_acc5 = evaluate(model, criterion, test_loader, device=device, header='Test:')
        if epoch >= args.prune_warmup and ((epoch - args.prune_warmup) % args.prune_interval == 0):


            # 先更新 mask（决定“将剪哪些通道”）
            mymanager.update_masks(model, args.alpha, args.beta)



            # 真正让 mask 生效（剪权重）
            mymanager.do_masks(model)


            '''
            mymanager.update_masks(model,args.alpha,args.beta) #alpha is the 1-(p+q) in the paper, beta id the q in the paper
            mymanager.do_masks(model)
            '''
            mymanager.compute_prune()

            mymanager.save_csv()
            mymanager.reset_zeros()
        else:
            print(f"[Warmup] epoch {epoch} < prune_warmup {args.prune_warmup}: skip pruning")

        
        
        writer.add_scalar('test_accuracy', test_acc1, epoch )
        if te_tb_writer is not None:
            if utils.is_main_process():
                te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)
        print(test_acc1)
        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            test_acc5_at_max_test_acc1 = test_acc5
            save_max = True
            best_name = f'vgg16_{args.dataset}_best.pth.tar'

            save_path = os.path.join(args.output_dir, best_name)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': max_test_acc1,
                'optimizer': optimizer.state_dict(),
            }, save_path)
            #print('saved')
            if epoch >= args.prune_warmup and len(mymanager.masks) > 0:
                mymanager.save_csv_max()

        print(max_test_acc1)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(output_dir)

        print('Training time {}'.format(total_time_str), 'max_test_acc1', max_test_acc1,
              'test_acc5_at_max_test_acc1', test_acc5_at_max_test_acc1)




    # ===== 训练结束后，执行一次自适应 REC =====
    if args.adaptive_rec and len(mymanager.masks) > 0:
        print('\n===== Start adaptive final reconstruction =====')



        if args.teacher_path != '':
            print(f"[Info] Loading teacher from: {args.teacher_path}")
            teacher = load_teacher_model(
                teacher_path=args.teacher_path,
                device=device,
                num_classes=num_classes,
                args=args
            )
        else:
            print("[Info] No teacher_path provided, fallback to current model snapshot.")
            teacher = copy.deepcopy(model).to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)

        calib_iter = _make_calib_iter(train_loader, args.ptp_calib_batches)

        adaptive_final_reconstruction(
            model=model,
            teacher=teacher,
            manager=mymanager,
            calib_loader=calib_iter,
            device=device,
            criterion=criterion,
            iters=args.adaptive_rec_iters,
            lr=args.ptp_lr,
            lambda_feat=args.adaptive_rec_lambda,
            topk=args.adaptive_rec_topk,
            use_amp=args.amp
        )

        print('===== Adaptive final reconstruction finished =====')

        final_test_loss, final_test_acc1, final_test_acc5 = evaluate(
            model, criterion, test_loader, device=device, header='Adaptive-REC Test:'
        )
        print('After adaptive reconstruction:', final_test_acc1, final_test_acc5)

    if args.adaptive_rec and len(mymanager.masks) > 0:
        adaptive_save_path = os.path.join(args.output_dir, 'adaptive_rec_final.pth.tar')
        torch.save({
            'state_dict': model.state_dict(),
            'acc1': final_test_acc1,
            'acc5': final_test_acc5,
        }, adaptive_save_path)




    ###########
    # ===== 保存最终结构 =====
    mymanager.save_final_mask()
    mymanager.save_cfg()



    # ===== 1) Connection (保留你的原统计，方便参考) =====
    nonzero_w, total_w, conn_percent = compute_connection_percent(model)

    # ===== 2) 读取压缩后 cfg =====
    cfg_path = os.path.join(args.output_dir, "cfg.txt")
    cfg_channels = load_cfg_channels(cfg_path)

    print("cfg_channels:", cfg_channels)
    print("len(cfg_channels):", len(cfg_channels))
    print("cfg_channels:", cfg_channels)

    if args.dataset == 'cifar10':
        num_classes_for_stats = 10
        input_size_for_stats = 32
    elif args.dataset == 'cifar100':
        num_classes_for_stats = 100
        input_size_for_stats = 32
    elif args.dataset == 'tiny_imagenet':
        num_classes_for_stats = 200
        input_size_for_stats = 64
    else:
        # dvscifar10 / others
        num_classes_for_stats = num_classes
        input_size_for_stats = 32

    # ===== 3) 压缩后参数量 =====
    compact_params = compute_compact_vgg_params(
        cfg_channels,
        num_classes=num_classes_for_stats
    )

    # ===== 4) 压缩后 spike rate =====
    spike_rates = calibrate_spike_rates(
        model=model,
        manager=mymanager,
        data_loader=test_loader,
        device=device,
        num_batches=10
    )

    # 这里要求和13层卷积对齐
    if len(spike_rates) != 13:
        print(f"[Warning] spike_rates length = {len(spike_rates)}, cfg_channels length = {len(cfg_channels)}")

    n = min(len(spike_rates), len(cfg_channels), 13)
    spike_rates = spike_rates[:n]
    cfg_channels_for_syn = cfg_channels[:n]

    # 只有在长度足够时才计算 SynOps
    if n == 13:
        compact_synops = compute_vgg_synops(
            cfg_channels_for_syn,
            spike_rates,
            T=args.T,
            input_size=input_size_for_stats,
            num_classes=num_classes_for_stats
        )
    else:
        compact_synops = -1
        print("[Warning] Cannot compute compact_synops exactly because n != 13")

    print("============== Final Compression Statistics ==============")
    print(f"Connection (%):    {conn_percent:.2f}")
    print(f"Compact Params:    {compact_params / 1e6:.4f} M")

    if compact_synops >= 0:
        print(f"Compact SynOps:    {compact_synops / 1e3:.4f} K")
    else:
        print("Compact SynOps:    unavailable")

    stats = {
        "connection_percent": conn_percent,
        "nonzero_weights": nonzero_w,
        "total_weights": total_w,
        "compact_params": compact_params,
        "compact_synops": compact_synops
    }



    df = pd.DataFrame([stats])
    df.to_csv(os.path.join(args.output_dir, "model_stats.csv"), index=False)






def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='./download/tiny-imagenet-200', help='dataset')

    parser.add_argument('--model', default='spiking_resnet18', help='model')
    parser.add_argument('--device', default='cuda:1', help='device')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=320, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')  
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD. Adam will not use momentum')
    parser.add_argument('--wd', '--weight_decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--beta', default=0.1, type=float)

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'dvscifar10', 'tiny_imagenet'],
                        help='choose dataset: cifar10 or cifar100')
    parser.add_argument('--frames-number', default=20, type=int,
                        help='frames number for DVS-CIFAR10')


    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=4, type=int, help='simulation steps')
    parser.add_argument('--adam', action='store_true',
                        help='Use Adam. The default optimizer is SGD.')

    parser.add_argument('--cos_lr_T', default=320, type=int,
                        help='T_max of CosineAnnealingLR.')
    parser.add_argument('--connect_f', default='ADD', type=str, help='spike-element-wise connect function')
    parser.add_argument('--zero_init_residual', action='store_true', help='zero init all residual blocks')
    parser.add_argument('--accum-steps', default=1, type=int,
                        help='gradient accumulation steps. effective batch = batch_size * accum_steps')

    parser.add_argument('--ptp', action='store_true',
                        help='enable two-stage reconstruction (IA + REC) around pruning')
    parser.add_argument('--ptp-calib-batches', default=10, type=int,
                        help='num of batches for calibration set each epoch')


    parser.add_argument('--ptp-ia-iters', default=10, type=int)
    parser.add_argument('--ptp-rec-iters', default=10, type=int)
    parser.add_argument('--ptp-reg', default=0.02, type=float)
    parser.add_argument('--ptp-inc', default=0.02, type=float)
    parser.add_argument('--ptp-lr', default=1e-3, type=float)





    parser.add_argument('--prune-warmup', default=0, type=int,
                        help='warmup epochs before pruning starts')
    parser.add_argument('--prune-interval', default=1, type=int,
                        help='apply pruning every N epochs after warmup')
    parser.add_argument('--scheduler', default='step', choices=['step', 'cosine'],
                        help='lr scheduler: step or cosine')


    parser.add_argument('--adaptive-rec', action='store_true',
                        help='run adaptive reconstruction once after full training')
    parser.add_argument('--adaptive-rec-iters', default=20, type=int)
    parser.add_argument('--adaptive-rec-topk', default=3, type=int)
    parser.add_argument('--adaptive-rec-lambda', default=1.0, type=float)
    parser.add_argument('--teacher_path', default='', type=str, help='path to teacher checkpoint')




    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

'''

tensorboard --logdir=

pip install torch
 pip install torchvision
pip install spikingjelly==0.0.0.0.12
 pip install tensorboardX

无IA
python m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cos_lr_T 320 --model sew_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --connect_f ADD --T 4 --lr 0.1 --epoch 320 --data-path /raid/wfang/imagenet

python train.py --cos_lr_T 320 --model spiking_resnet18 -b 32 --output-dir ./logs --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch 320 --data-path /raid/wfang/imagenet --device cuda:0 --zero_init_residual


python train.py  --output-dir ./logs --tb  --amp  --T 4 --lr 0.1 --epoch 320  --device cuda:0 --zero_init_residual --amp

#tiny-imagenet

mkdir -p ./download
cd ./download
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip


nohup python train.py --dataset tiny_imagenet --data-path ./download/tiny-imagenet-200 --batch-size 16 --lr 0.01 --device cuda:0 --output-dir ./tiny_imagenet_trainlog --tb --accum-steps 4 > train_tiny_imagenet.log 2>&1 &

#cifar10dvs

nohup python train.py --dataset dvscifar10 --frames-number 10 --batch-size 4 --accum-steps 16 --lr 0.01 --device cuda:0 --output-dir ./cifar10dvs_trainlog --tb --amp --zero_init_residual > train_dvscifar10dvs.log 2>&1 &


有IA

#cifar10

nohup python train.py --dataset cifar10  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar10_IA_trainlog --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > cifar10_IA_train.log 2>&1 &


#cifar100
nohup python train.py --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_IA_log --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > cifar100_IA_train.log 2>&1 &

nohup python train.py --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_IA_log --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > cifar100_IA_train.log 2>&1 &

uniquness + gate + 调参

nohup python train.py --adam --scheduler cosine --lr 1e-3 --wd 1e-4 --epochs 200 --prune-warmup 40 --prune-interval 1 --dataset cifar100 --batch-size 64 --device cuda:0 --output-dir ./cifar100_uniqueness_gate_adam_cosine_log --tb --amp --zero_init_residual > cifar100_uniqueness_gate_adam_cosine_train.log 2>&1 &


#cifar10dvs

nohup python train.py --dataset dvscifar10 --frames-number 10 --batch-size 4 --accum-steps 16 --lr 0.01 --device cuda:0 --output-dir ./cifar10dvs_IA_trainlog --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > dvscifar10_IA_train.log 2>&1 &

nohup python train.py  --prune-interval 1 --epochs 200 --prune-warmup 0 --dataset dvscifar10 --frames-number 10 --batch-size 4 --accum-steps 16  --lr 0.01 --device cuda:0 --output-dir ./cifar10dvs_unqueness_gate_regrowth_log --tb --amp --zero_init_residual > cifar10dvs_uniquness_gate_regrowth_train.log 2>&1 &


更改评估方法，无IA
nohup python train.py --prune-interval 5 --epochs 200 --prune-warmup 20 --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_snr_log --tb --amp --zero_init_residual > cifar100_snr_train.log 2>&1 &

uniqueness + gate 无IA
nohup python train.py  --prune-interval 1 --epochs 150 --prune-warmup 0 --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_unqueness_gate_log --tb --amp --zero_init_residual > cifar100_uniquness_gate_train.log 2>&1 &

uniqueness + gate 有IA
nohup python train.py  --prune-interval 1 --epochs 200 --prune-warmup 0 --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_unqueness_gate_IA_log --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > cifar100_uniquness_IA_gate_train.log 2>&1 &

uniqueness + gate + regrowth
nohup python train.py  --prune-interval 1 --epochs 200 --prune-warmup 0 --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_unqueness_gate_regrowth_log --tb --amp --zero_init_residual > cifar100_uniquness_gate_regrowth_train.log 2>&1 &

uniqueness + gate a60b10 有IA
nohup python train.py  --prune-interval 1 --epochs 200 --prune-warmup 30 --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_unqueness_gate_IA_a60b10_log --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > cifar100_uniquness_gate_IA_a60b10_train.log 2>&1 &


#tiny-imagenet


nohup python train.py --dataset tiny_imagenet --data-path ./download/tiny-imagenet-200 --batch-size 16 --accum-steps 4 --lr 0.01 --device cuda:0 --output-dir ./tiny_imagenet_trainlog --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > tiny_imagenet_IA_train.log 2>&1 &
nohup python train.py --dataset tiny_imagenet --data-path ./download/tiny-imagenet-200 --batch-size 16 --accum-steps 4 --lr 0.01 --device cuda:0 --output-dir ./tiny_imagenet_trainlog --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > tiny_imagenet_IA_train.log 2>&1 &

调整评估方法
nohup python train.py --dataset tiny_imagenet --data-path ./download/tiny-imagenet-200 --batch-size 16 --accum-steps 4 --lr 0.01 --device cuda:0 --output-dir ./tiny_imagenet_update12_38_4 --tb --amp --zero_init_residual > tiny_imagenet_update12_38_4.log 2>&1 &

nohup python train.py --dataset tiny_imagenet --data-path ./download/tiny-imagenet-200 --batch-size 16 --accum-steps 4 --lr 0.01 --device cuda:0 --output-dir ./tiny_imagenet_update12_IA_38_4 --epochs 200 --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > tiny_imagenet_update12_38_4.log 2>&1 &



nohup python train.py --prune-interval 1 --epochs 200 --prune-warmup 0 --dataset tiny_imagenet --data-path ./download/tiny-imagenet-200 --batch-size 16 --accum-steps 4 --lr 0.01 --device cuda:0 --output-dir ./tiny_imagenet_trainlog --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 > tiny_imagenet_IA_train.log 2>&1 &

# uniqueness+gate+regrowth
nohup python train.py --prune-interval 1 --epochs 200 --prune-warmup 0 --dataset tiny_imagenet --data-path ./download/tiny-imagenet-200 --batch-size 16 --accum-steps 4 --lr 0.01 --device cuda:0 --output-dir ./tiny_imagenet_uniqueness_gate_regrowth_trainlog --tb --amp --zero_init_residual  > tiny_imagenet_uniqueness_gate_regrowth_train.log 2>&1 &


更改剪枝率
#cifar100 0.6 0.1 无IA
nohup python train.py --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_a60b10_log --tb --amp --zero_init_residual > cifar100_a60b10_train.log 2>&1 &
#cifar100 0.6 0.1 IA
nohup python train.py --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_uniquneness_gate_a60b10_IA_log --tb --amp --zero_init_residual --ptp --ptp-calib-batches 10 --ptp-ia-iters 20 --ptp-rec-iters 30 --ptp-reg 5e-5 --ptp-inc 5e-5 --ptp-lr 1e-4 --alpha 0.6 > cifar100_a60b10_IA_train.log 2>&1 &

#调整评估方法 3.8
nohup python train.py --dataset cifar100  --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_update38 --tb --amp --zero_init_residual > cifar100_update38.log 2>&1 &



nohup python train.py --dataset cifar100 --adam --scheduler cosine --lr 1e-3 --wd 1e-4 --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_update38_cosine --tb --amp --zero_init_residual > cifar100_update38_cosine.log 2>&1 &


nohup python train.py --dataset cifar100 --adam --scheduler cosine --lr 1e-3 --wd 1e-4 --batch-size 64  --lr 0.01 --device cuda:0 --output-dir ./cifar100_update38_cosine --tb --amp --zero_init_residual > cifar100_update38_cosine.log 2>&1 &

nohup python train.py  --dataset cifar100 -b 64 --output-dir ./cifar100_origin_39 --tb --print-freq 4096 --amp --cache-dataset --T 4 --lr 0.1 --epoch  200 --alpha 1.0 --beta 0 --device cuda:0 --zero_init_residual   > cifar100_origin_39.log 2>&1 &


'''
