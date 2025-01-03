from __future__ import print_function

import sys
import time
import math
import numpy as np
import random
import seaborn as sns

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
from torch import autocast
from torch import autocast

from models import PCRLv2, Cluster
from tools import adjust_learning_rate, AverageMeter, swav_loss, roi_align_intersect


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


def train_2d(args, data_loader, run_dir, writer=None):

    if 'cluster' in args.model:
        # Generate colors for cluster masks
        palette = sns.color_palette(palette='bright', n_colors=args.k)
        colors = torch.Tensor([list(color) for color in palette]).cpu()  # cpu because we apply it on a detached tensor later

    torch.backends.cudnn.deterministic = True

    train_loader = data_loader['train']
    
    # Create model and optimizer
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
    model = nn.DataParallel(model)

    criterion = nn.MSELoss()
    cosine = nn.CosineSimilarity()
    if not args.cpu:
        criterion = criterion.cuda()
        cosine = cosine.cuda()

    for epoch in range(0, args.epochs + 1):

        # TRAINING

        adjust_learning_rate(epoch, args, optimizer)
        print("==> Training...", flush=True)

        time1 = time.time()
        
        if args.model == 'pcrlv2':
            loss, prob, total_loss, writer = train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer)
        elif 'cluster' in args.model:
            loss, prob, total_loss, writer = train_cluster_inner(args, epoch, train_loader, model, optimizer, writer, colors)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1), flush=True)
        
        if args.tensorboard:
            writer.add_scalar('loss/train', total_loss, epoch)  # Write train loss on tensorboard
        
        # Save model
        if epoch % 100 == 0 or epoch == 240:
            # saving the model
            print('==> Saving...', flush=True)
            state = {'opt': args, 'state_dict': model.module.model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}

            save_file = run_dir + '.pt'
            torch.save(state, save_file)

            # Help release GPU memory
            del state

        if not args.cpu:
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
    for idx, (input1, input2, gt1, gt2, _, _, local_views, _) in enumerate(train_loader):
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
                data_time=data_time, c2l_loss=loss_meter, mg_loss=mg_loss_meter, prob=prob_meter), flush=True)

    return mg_loss_meter.avg, prob_meter.avg, total_loss_meter.avg, writer


def train_pcrlv2_2d(args, data_loader, run_dir, writer=None):
    train_2d(args, data_loader, run_dir, writer=writer)


def train_cluster_inner(args, epoch, train_loader, model, optimizer, writer, colors):
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
    
    for idx, (input1, input2, _, _, crop1_coords, crop2_coords, _, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x1 = input1.float()
        x2 = input2.float()

        if not args.cpu:
            x1 = x1.cuda()
            x2 = x2.cuda()
            crop1_coords = crop1_coords.cuda()
            crop2_coords = crop2_coords.cuda()

        # Convert 3D input to 2D
        B, C, H, W, D = x1.shape
        x1 = x1.permute(0,4,1,2,3).reshape(B*D,C,H,W)  # B x C x H x W x D -> B*D x C x H x W
        x2 = x2.permute(0,4,1,2,3).reshape(B*D,C,H,W)
        crop1_coords = torch.repeat_interleave(crop1_coords[:,:4],repeats=D,dim=0)  # Repeat crop window for each slice and ignore z dim of crop window
        crop2_coords = torch.repeat_interleave(crop2_coords[:,:4],repeats=D,dim=0)

        device_type = 'cpu' if args.cpu else 'cuda'
        with autocast(device_type=device_type):  # Run in mixed-precision

            # Get cluster predictions from student U-Net
            pred1 = model.module(x1)
            pred2 = model.module(x2)
            
            # Convert to probabilities
            pred1 = pred1.softmax(2)
            pred2 = pred2.softmax(2)

        # ROI-align crop intersection with cluster assignment intersection
        roi_pred1, roi_pred2, roi_gt1, roi_gt2 = roi_align_intersect(pred1, pred2, gt1, gt2, crop1_coords, crop2_coords)

        # SwAV Loss for current scale
        loss1 = swav_loss(roi_gt1, roi_gt2, roi_pred1, roi_pred2)

        # Plot predictions on tensorboard
        with torch.no_grad():
            b_idx = 0
            if args.vis and idx==b_idx and epoch % 10 == 0:

                # Select images
                img_idx = D//2 # TODO: D//2
                m_idx = 0
                c_idx = 0
                in1 = x1[img_idx,m_idx,:,:].unsqueeze(0)
                in2 = x2[img_idx,m_idx,:,:].unsqueeze(0)
                pred1 = pred1[img_idx,:,:,:].argmax(dim=0).unsqueeze(0)  # Take only hard cluster assignment (argmax)
                pred2 = pred2[img_idx,:,:,:].argmax(dim=0).unsqueeze(0)
                gt1 = gt1[img_idx,:,:,:].argmax(dim=0).unsqueeze(0)
                gt2 = gt2[img_idx,:,:,:].argmax(dim=0).unsqueeze(0)

                # Min-max norm input images
                in1 = (in1 - in1.min())/(in1.max() - in1.min())
                in2 = (in2 - in2.min())/(in2.max() - in2.min())

                # Send to cpu
                in1 = in1.cpu().detach()
                in2 = in2.cpu().detach()
                pred1 = pred1.cpu().detach()
                pred2 = pred2.cpu().detach()
                gt1 = gt1.cpu().detach()
                gt2 = gt2.cpu().detach()

                # Give color to each cluster in cluster masks
                pred1 = pred1.repeat((3,1,1)).permute(1,2,0).float()  # Convert to RGB and move channel dim to the end
                pred2 = pred2.repeat((3,1,1)).permute(1,2,0).float()
                gt1 = gt1.repeat((3,1,1)).permute(1,2,0).float()
                gt2 = gt2.repeat((3,1,1)).permute(1,2,0).float()
                for c in range(colors.shape[0]):
                    pred1[pred1[:,:,0] == c] = colors[c]
                    pred2[pred2[:,:,0] == c] = colors[c]
                    gt1[gt1[:,:,0] == c] = colors[c]
                    gt2[gt2[:,:,0] == c] = colors[c]
                pred1 = pred1.permute(2,1,0)
                pred2 = pred2.permute(2,1,0)
                gt1 = gt1.permute(2,0,1)
                gt2 = gt2.permute(2,0,1)

                # Pad images for better visualization                
                in1 = f.pad(in1.unsqueeze(0),(2,1,2,2),value=1)
                in2 = f.pad(in2.unsqueeze(0),(1,2,2,2),value=1)
                pred1 = f.pad(pred1.unsqueeze(0),(2,1,2,2),value=1)
                pred2 = f.pad(pred2.unsqueeze(0),(1,2,2,2),value=1)
                gt1 = f.pad(gt1.unsqueeze(0),(2,1,2,2),value=1)
                gt2 = f.pad(gt2.unsqueeze(0),(1,2,2,2),value=1)

                # Combine crops and save in tensorboard
                in_img = torch.cat((in1,in2),dim=3).squeeze(0).cpu().detach().numpy()
                pred_img = torch.cat((pred1,pred2),dim=3).squeeze(0).cpu().detach().numpy()
                gt_img = torch.cat((gt1,gt2),dim=3).squeeze(0).cpu().detach().numpy()

                in_img_name = 'img/train/raw' 
                pred_img_name = f'img/train/pred'
                gt_img_name = f'img/train/gt'

                writer.add_image(in_img_name, img_tensor=in_img, global_step=epoch*len(train_loader)+idx, dataformats='CHW')
                writer.add_image(pred_img_name, img_tensor=pred_img, global_step=epoch*len(train_loader)+idx, dataformats='CHW')   
                writer.add_image(gt_img_name, img_tensor=gt_img, global_step=epoch*len(train_loader)+idx, dataformats='CHW')

        # TODO: add the other losses later
        loss1 = loss1 / D 
        loss2 = torch.tensor(0) / D
        loss4 = 0 / D
        local_loss = 0  # / DL

        # Total Loss
        # TODO: add the other losses later
        loss = loss1

        # Backward
        optimizer.zero_grad()
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
                data_time=data_time, c2l_loss=loss_meter, mg_loss=mg_loss_meter, prob=prob_meter), flush=True)

    return mg_loss_meter.avg, prob_meter.avg, total_loss_meter.avg, writer


def train_cluster_2d(args, data_loader, run_dir, writer=None):
    train_2d(args, data_loader, run_dir, writer=writer)
