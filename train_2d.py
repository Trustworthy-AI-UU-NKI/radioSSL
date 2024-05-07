from __future__ import print_function

import os
import sys
import time
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
import random
from tools import adjust_learning_rate, AverageMeter
from models import PCRLv2, Cluster
from torch_kmeans import KMeans

try:
    from apex import amp, optimizers
except ImportError:
    pass


# from koila import LazyTensor, lazy

def Normalize(x):
    norm_x = x.pow(2).sum(1, keepdim=True).pow(1. / 2.)
    x = x.div(norm_x)
    return x


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def mixup_data(x, alpha=1.0, index=None, lam=None, ):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if lam is None:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = lam

    lam = max(lam, 1 - lam)
    batch_size = x.size()[0]
    if index is None:
        index = torch.randperm(batch_size).cuda()
    else:
        index = index

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, lam, index


def cos_loss(cosine, output1, output2):
    index = random.randint(0, len(output1) - 1)
    sample1 = output1[index]
    sample2 = output2[index]
    loss = -(cosine(sample1[1], sample2[0].detach()).mean() + cosine(sample2[1],
                                                                     sample1[0].detach()).mean()) * 0.5
    return loss, index


def train_2d(args, data_loader, run_dir, out_channel=3, writer=None):
    train_loader = data_loader['train']
    
    # create model and optimizer
    if args.model == 'pcrlv2':
        model = PCRLv2()
    elif 'cluster' in args.model:
        model = Cluster(n_clusters=args.k, seed=args.seed)

    if not args.cpu:
        model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    cosine = nn.CosineSimilarity()
    if not args.cpu:
        criterion = criterion.cuda()
        cosine = cosine.cuda()
        cudnn.benchmark = True

    for epoch in range(0, args.epochs + 1):


        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        if args.model == 'pcrlv2':
            loss, prob, total_loss = train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer)
        elif 'cluster' in args.model:
            loss, prob, total_loss = train_cluster_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        if args.tensorboard:
            writer.add_scalar('loss/train', total_loss, epoch)  # Write train loss on tensorboard

        
        # save model
        if epoch % 100 == 0 or epoch == 240:
            # saving the model
            print('==> Saving...')
            state = {'opt': args, 'state_dict': model.module.model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}

            save_file = run_dir + '.pt'
            torch.save(state, save_file)

            # help release GPU memory
            del state
        torch.cuda.empty_cache()


def train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer):
    """
    one epoch training for instance discrimination
    """

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    mg_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    end = time.time()
    for idx, (input1, input2, gt1, gt2, _, _, local_views) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x1 = input1.float()
        x2 = input2.float()
        gt1 = gt1.float()

        if not args.cpu:
            x1 = x1.cuda()
            x2 = x2.float()
            gt1 = gt1.cuda()

        # Convert 3D global views to 2D
        B, M, H, W, D = x1.shape
        _, C, _, _, _ = gt1.shape 
        x1 = x1.permute(0,4,1,2,3).reshape(B*D,M,H,W)  # B x M x H x W x D -> B*D x M x H x W
        x2 = x2.permute(0,4,1,2,3).reshape(B*D,M,H,W)
        gt1 = gt1.permute(0,4,1,2,3).reshape(B*D,M,H,W)
        gt2 = gt2.permute(0,4,1,2,3).reshape(B*D,M,H,W)

        decoder_outputs1, mask1, middle_masks1 = model(x1)
        decoder_outputs2, mask2, _ = model(x2)

        loss2, index2 = cos_loss(cosine, decoder_outputs1, decoder_outputs2)
        loss2 = loss2 / D

        local_loss = 0.0

        local_input = torch.cat(local_views, dim=0) 

        # Convert 3D local views to 2D
        BL, _, HL, WL, DL = local_input.shape # BL = 6 * B 
        local_input = local_input.permute(0,4,1,2,3).reshape(BL*DL,M,HL,WL)  # 6 * B * DL, 3, 96, 96

        local_views_outputs, _, _ = model(local_input, local=True) # 4 * 2 * [6 * B * DL, 3, 96, 96]
        local_views_outputs = [torch.stack(t) for t in local_views_outputs]

        # Because global and local views have diff num of slices so diff batch sizes (B*D and B*DL), we only take the slices of the local view from the global one
        decoder_outputs1 = [[t[0][:B*DL,:], t[1][:B*DL,:]] for t in decoder_outputs1]
        decoder_outputs2 = [[t[0][:B*DL,:], t[1][:B*DL,:]] for t in decoder_outputs2]

        for i in range(len(local_views)):
            local_views_outputs_tmp = [t[:, B * DL * i: B * DL * (i + 1)] for t in local_views_outputs]  # We use B * DL and not BL * BD because BL = B*6 and we iterate over the 6 local views
            loss_local_1, _ = cos_loss(cosine, decoder_outputs1, local_views_outputs_tmp)
            loss_local_2, _ = cos_loss(cosine, decoder_outputs2, local_views_outputs_tmp)
            local_loss += loss_local_1
            local_loss += loss_local_2

        local_loss = local_loss / (DL * 2 * len(local_views))
        loss1 = criterion(mask1, gt1) / D
        beta = 0.5 * (1. + math.cos(math.pi * epoch / 240))
        loss4 = beta * criterion(middle_masks1[index2], gt1) / D
        
        # Total Loss
        loss = loss1 + loss2 + loss4 + local_loss

        # Backward
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Meters
        mg_loss_meter.update(loss1.item(), B)
        loss_meter.update(loss2.item(), B)
        prob_meter.update(local_loss, B)
        total_loss_meter.update(loss.item(), B)
        if not args.cpu:
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # Print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'cos_loss {c2l_loss.val:.3f} ({c2l_loss.avg:.3f})\t'
                  'mg loss {mg_loss.val:.3f} ({mg_loss.avg:.3f})\t'
                  'local loss {prob.val:.3f} ({prob.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, c2l_loss=loss_meter, mg_loss=mg_loss_meter, prob=prob_meter))
            sys.stdout.flush()

    return mg_loss_meter.avg, prob_meter.avg, total_loss_meter.avg


def train_pcrlv2_2d(args, data_loader, run_dir, out_channel=3, writer=None):
    train_2d(args, data_loader, run_dir, out_channel=out_channel, writer=writer)


def train_cluster_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer):
    """
    one epoch training for instance discrimination
    """

    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    mg_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    end = time.time()
    for idx, (input1, input2, _, _, crop1_coords, crop2_coords, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x1 = input1.float()
        x2 = input2.float()

        if not args.cpu:
            x1 = x1.cuda()
            x2 = x2.cuda()
            crop1_coords = crop1_coords.cuda()
            crop2_coords = crop2_coords.cuda()

        # Convert 3D input to 2D
        B, M, H, W, D = x1.shape
        x1 = x1.permute(0,4,1,2,3).reshape(B*D,M,H,W)  # B x M x H x W x D -> B*D x M x H x W
        x2 = x2.permute(0,4,1,2,3).reshape(B*D,M,H,W)

        # Get cluster predictions from student U-Net
        print(x1.device,x2.device)
        pred1 = model.module(x1)
        pred2 = model.module(x2)

        # Get upsampled features from teacher DINO ViT16 encoder
        with torch.no_grad():
            feat1 = model.module.featup_upsampler(x1.repeat(1,3,1,1))  # Requires 3 channels
            feat2 = model.module.featup_upsampler(x2.repeat(1,3,1,1))
        
        # Flatten spatial dimensions to get feature vectors for each pixel
        feat_vec1 = feat1.permute(0,2,3,1).flatten(0,2)  # B*D x C' x H' x W' -> B*D*H'*W' x C'
        feat_vec2 = feat2.permute(0,2,3,1).flatten(0,2)  # B*D x C' x H' x W' -> B*D*H'*W' x C'

        # Log images on tensorboard
        if args.tensorboard and args.vis:
            in_img = x1[0].cpu().detach().numpy()
            pred_img = pred1[0].cpu().detach().numpy()
            dino_img = feat1[0].cpu().detach().numpy()

            in_img_name = 'img/train/raw' 
            pred_img_name = f'img/train/pred'
            dino_img_name = f'img/train/dino'

            writer.add_image(in_img_name, img_tensor=in_img, global_step=epoch, dataformats='CHW')
            writer.add_image(pred_img_name, img_tensor=pred_img, global_step=epoch, dataformats='CHW')   
            writer.add_image(dino_img_name, img_tensor=dino_img, global_step=epoch, dataformats='CHW')

        # Perform K-Means on teacher feature vectors
        model.kmeans.fit_predict(x=feat_vec2)
        print()

        


        # TODO: add the other losses later
        loss1 = 0 / D  #TODO fill in
        loss2 = torch.tensor(0) / D
        loss4 = 0 / D
        local_loss = 0  # / DL

        # Total Loss
        # TODO: add the other losses later
        loss = loss1

        # Backward
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # Meters
        mg_loss_meter.update(loss1.item(), B)
        loss_meter.update(loss2.item(), B)
        prob_meter.update(local_loss, B)
        total_loss_meter.update(loss.item(), B)
        if not args.cpu:
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # Print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'cos_loss {c2l_loss.val:.3f} ({c2l_loss.avg:.3f})\t'
                  'mg loss {mg_loss.val:.3f} ({mg_loss.avg:.3f})\t'
                  'local loss {prob.val:.3f} ({prob.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, c2l_loss=loss_meter, mg_loss=mg_loss_meter, prob=prob_meter))
            sys.stdout.flush()

    return mg_loss_meter.avg, prob_meter.avg, total_loss_meter.avg


def train_cluster_2d(args, data_loader, run_dir, out_channel=3, writer=None):
    train_2d(args, data_loader, run_dir, out_channel=out_channel, writer=writer)
