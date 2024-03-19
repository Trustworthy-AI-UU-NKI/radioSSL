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
from utils import adjust_learning_rate, AverageMeter
from models import PCRLv2

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


def train_pcrlv2(args, data_loader, run_dir, out_channel=3, writer=None):
    train_loader = data_loader['train']
    # create model and optimizer
    model = PCRLv2()

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

        loss, prob, total_loss = train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        if args.tensorboard:
            writer.add_scalar('loss/train', total_loss, epoch)  # Write train loss on tensorboard

        
        # save model
        if epoch % 100 == 0 or epoch == 240:
            # saving the model
            print('==> Saving...')
            state = {'opt': args, 'state_dict': model.module.model.encoder.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}

            save_file = run_dir + '.pt'
            torch.save(state, save_file)

            # help release GPU memory
            del state
        torch.cuda.empty_cache()


def cos_loss(cosine, output1, output2):
    index = random.randint(0, len(output1) - 1)
    sample1 = output1[index]
    sample2 = output2[index]
    loss = -(cosine(sample1[1], sample2[0].detach()).mean() + cosine(sample2[1],
                                                                     sample1[0].detach()).mean()) * 0.5
    return loss, index


def train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine):
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

        # Convert 3D input to 2D
        B, M, H, W, D = x1.shape
        _, C, _, _, _ = gt1.shape 
        x1 = x1.permute(0,4,1,2,3).reshape(B*D,M,H,W)  # B x M x H x W x D -> B*D x M x H x W
        x2 = x2.permute(0,4,1,2,3).reshape(B*D,M,H,W)
        gt1 = gt1.permute(0,4,1,2,3).reshape(B*D,M,H,W)
        gt2 = gt2.permute(0,4,1,2,3).reshape(B*D,M,H,W)

        decoder_outputs1, mask1, middle_masks1 = model(x1)
        decoder_outputs2, mask2, _ = model(x2)

        loss2, index2 = cos_loss(cosine, decoder_outputs1, decoder_outputs2)
        local_loss = 0.0

        local_input = torch.cat(local_views, dim=0) 

        # Convert 3D input to 2D
        BL, _, HL, WL, DL = local_input.shape
        local_input = local_input.permute(0,4,1,2,3).reshape(BL*DL,M,HL,WL)  # 6 * B * D, 3, 96, 96

        local_views_outputs, _, _ = model(local_input, local=True) # 5 * 2 * [6 * B * D, 3, 96, 96]
        local_views_outputs = [torch.stack(t) for t in local_views_outputs]

        for i in range(len(local_views)):
            local_views_outputs_tmp = [t[:, B * D * i: B * D * (i + 1)] for t in local_views_outputs]
            loss_local_1, _ = cos_loss(cosine, decoder_outputs1, local_views_outputs_tmp)
            loss_local_2, _ = cos_loss(cosine, decoder_outputs2, local_views_outputs_tmp)
            local_loss += loss_local_1
            local_loss += loss_local_2
        local_loss = local_loss / (2 * len(local_views))
        loss1 = criterion(mask1, gt1)
        beta = 0.5 * (1. + math.cos(math.pi * epoch / 240))
        loss4 = beta * criterion(middle_masks1[index2], gt1)
        
        # Total Loss
        loss = loss1 + loss2 + local_loss + loss4

        # Backward
        optimizer.zero_grad()
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # clip_value = 10
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        # Meters
        mg_loss_meter.update(loss1.item(), bsz)
        loss_meter.update(loss2.item(), bsz)
        prob_meter.update(local_loss, bsz)
        total_loss_meter.update(loss.item(), bsz)
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

    return mg_loss_meter.avg, prob_meter.avg, total_loss_meter.avg, writer
