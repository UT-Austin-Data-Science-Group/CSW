# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from slicers import *
import cfg
import models
import datasets
from functions import train_sw, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
import numpy as np
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    gen_net = eval('models.'+args.model+'.Generator')(args=args).cuda()
    dis_net = eval('models.'+args.model+'.Discriminator')(args=args).cuda()

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/fid_stats_stl10.npz'
    elif args.dataset.lower() == 'celeba':
        fid_stat = 'fid_stat/fid_stats_celeba.npz'
    elif args.dataset.lower() == 'celebahq':
        fid_stat = 'fid_stat/fid_stats_celebahq.npz'
    elif args.dataset.lower() == 'lsun_church':
        fid_stat = 'fid_stat/fid_stats_lsun_church.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)
    if(args.slice_type=='sw'):
        if args.dataset.lower() == 'stl10':
            slicer = Base_Slicer(d=4608,L=args.L).cuda()
        else:
            slicer = Base_Slicer(d=8192,L=args.L).cuda()
    elif (args.slice_type == 'gsw'):
        if (args.s_act == 'sigmoid'):
            s_act = nn.Sigmoid()
        if args.dataset.lower() == 'stl10':
            slicer = NonLinearBase_Slicer(d=4608, L=args.L,activation=s_act).cuda()
        else:
            slicer = NonLinearBase_Slicer(d=8192, L=args.L,activation=s_act).cuda()
    elif (args.slice_type == 'ncsw' or args.slice_type == 'ncswd' or args.slice_type == 'ncsws'):
        if(args.s_act=='sigmoid'):
            s_act=nn.Sigmoid()
        if args.dataset.lower() == 'stl10':
            slicer = NonLinearConvSlicer(L=args.L, ch=args.df_dim, bottom_width=6,type=args.slice_type,activation=s_act)
        else:
            slicer = NonLinearConvSlicer(L=args.L, ch=args.df_dim, bottom_width=8,type=args.slice_type,activation=s_act)
    else:
        if args.dataset.lower() == 'stl10':
            slicer = ConvSlicer(L=args.L, ch=args.df_dim, bottom_width=6,type=args.slice_type)
        else:
            slicer = ConvSlicer(L=args.L, ch=args.df_dim, bottom_width=8,type=args.slice_type)
    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    gen_avg_param = copy_params(gen_net)
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
#         if(args.last_act):
#             args.path_helper = set_log_dir(args.dataset.lower() + 'last_logs', args.exp_name)
#         else:
#             if(args.last_feature):
#                 args.path_helper = set_log_dir(args.dataset.lower()+'onlyl_logs', args.exp_name)
#             else:
        args.path_helper = set_log_dir(args.dataset.lower() + '_logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }

    # train loop
    fids=[]
    iscores=[]
    lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        train_sw(args, gen_net, dis_net, slicer,gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
              lr_schedulers)

        if epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict)
            iscores.append(inception_score)
            fids.append(fid_score)
            np.savetxt(args.path_helper['log_path'] + "/fid.csv", np.array(fids), delimiter=",")
            np.savetxt(args.path_helper['log_path'] + "/is.csv", np.array(iscores), delimiter=",")
            logger.info(f'Inception score: {inception_score}, FID score: {fid_score} || @ epoch {epoch}.')
            load_params(gen_net, backup_param)
            if fid_score < best_fid:
                best_fid = fid_score
                is_best = True
            else:
                is_best = False
        else:
            is_best = False

        avg_gen_net = deepcopy(gen_net)
        load_params(avg_gen_net, gen_avg_param)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper
        }, is_best, args.path_helper['ckpt_path'])
        del avg_gen_net


if __name__ == '__main__':
    main()
