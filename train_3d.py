"""
Training code for C2L

"""
from __future__ import print_function

import os
import io
import sys
import time
import math
import random
import PIL

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms.functional as t
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
from matplotlib import pyplot as plt

from models import PCRLv23d, Cluster3d, TraceWrapper
from utils import adjust_learning_rate, AverageMeter, sinkhorn, swav_loss, roi_align_intersect


try:
    from apex import amp, optimizers
except ImportError:
    pass


def Normalize(x):
    norm_x = x.pow(2).sum(1, keepdim=True).pow(1. / 2.)
    x = x.div(norm_x)
    return x


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)


def cos_loss(cosine, output1, output2):
    index = random.randint(0, len(output1) - 1)  # Because we select a feature map from a random scale
    sample1 = output1[index]
    sample2 = output2[index]
    loss = -(cosine(sample1[1], sample2[0].detach()).mean() + cosine(sample2[1],
                                                                     sample1[0].detach()).mean()) * 0.5
    return loss, index


def train_3d(args, data_loader, run_dir, out_channel=3, writer=None):

    if args.model == 'cluster':
        # Generate colors for cluster masks
        palette = sns.color_palette(palette='bright', n_colors=args.k)
        colors = torch.Tensor([list(color) for color in palette])

    torch.backends.cudnn.deterministic = True

    train_loader = data_loader['train']
    val_loader = data_loader['eval']

    # Create model and optimizer
    if args.model == 'pcrlv2':
        model = PCRLv23d(skip_conn=args.skip_conn)
    elif args.model == 'cluster':
        model = Cluster3d(n_clusters=args.k)
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

    grid_preds = []  # Clustering task predictions to visualize in grid (epoch X image) 

    for epoch in range(0, args.epochs + 1):

        # TRAINING

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        if args.model == 'pcrlv2':
            loss, prob, total_loss, writer = train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer)
        elif args.model == 'cluster':
            loss, prob, total_loss, writer = train_cluster_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer, colors)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if args.tensorboard:
            writer.add_scalar('loss/train', total_loss, epoch)  # Write train loss on tensorboard

        # VALIDATION (only for clustering task, just for visualization purposes)

        if args.model == 'cluster' and args.n == 'brats':  # TODO: currently only works for BraTS dataset
            print("==> validating...")
            row_preds = val_cluster_inner(args, epoch, val_loader, model, colors)
            grid_preds.extend(row_preds)

        # Save model
        if epoch % 100 == 0 or epoch == 240:
            print('==> Saving...')
            state = {'opt': args, 'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}

            save_file = run_dir + '.pt'
            torch.save(state, save_file)

            # Help release GPU memory
            del state

        if not args.cpu:
            torch.cuda.empty_cache()
    
        # Visualize grid of predictions for clustering task
        if args.model == 'cluster' and args.n == 'brats':  # TODO: Currently only works for BraTS dataset
            n_cols = min(10, args.b)
            n_rows = len(grid_preds) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15*(n_rows/n_cols)))
            for i, ax in enumerate(axes.flat):
                ax.imshow(grid_preds[i])
                ax.axis('off')  # Turn off axis labels
            plt.tight_layout()  # Adjust spacing between subplots
            
            # Save grid to buffer and then log on tensorboard
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            grid = PIL.Image.open(buf)
            grid = t.pil_to_tensor(grid)
            writer.add_image('img/val/grid', img_tensor=grid, global_step=epoch)

        
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

        B, C, H, W, D = input1.shape

        data_time.update(time.time() - end)

        bsz = input1.size(0)
        x1 = input1.float()  # Crop 1
        x2 = input2.float()  # Crop 2

        if not args.cpu:
            x1 = x1.cuda()
            x2 = x2.cuda()
            gt1 = gt1.cuda()
            gt2 = gt2.cuda()
            crop1_coords = crop1_coords.cuda()
            crop2_coords = crop2_coords.cuda()

        # Get predictions
        mask1, decoder_outputs1, middle_masks1 = model(x1)
        mask2, decoder_outputs2, _ = model(x2)

        loss2, index2 = cos_loss(cosine, decoder_outputs1, decoder_outputs2)
        local_loss = 0.0

        local_input = torch.cat(local_views, dim=0)  # 6 * bsz, 3, d, 96, 96
        _, local_views_outputs, _ = model(local_input, local=True)  # 4 * 2 * [6 * bsz, 3, d, 96, 96]
        local_views_outputs = [torch.stack(t) for t in local_views_outputs]
        
        for i in range(len(local_views)):
            local_views_outputs_tmp = [t[:, bsz * i: bsz * (i + 1)] for t in local_views_outputs]
            loss_local_1, _ = cos_loss(cosine, decoder_outputs1, local_views_outputs_tmp)
            loss_local_2, _ = cos_loss(cosine, decoder_outputs2, local_views_outputs_tmp)
            local_loss += loss_local_1
            local_loss += loss_local_2
        local_loss = local_loss / (2 * len(local_views))
        loss1 = criterion(mask1, gt1)
        beta = 0.5 * (1. + math.cos(math.pi * epoch / 240))
        loss4 = beta * criterion(middle_masks1[index2], gt1)  
        
        # Total Loss
        loss = loss1 + loss2 + loss4 + local_loss

        # Backward
        if loss > 1000 and epoch > 10:
            print('skip the step')
            continue
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
        total_loss_meter.update(loss, bsz)
        if not args.cpu:
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # if args.tensorboard:
        #     if epoch == 0:  # Only on the first iteration, write model graph on tensorboard
        #         model_wrapper = TraceWrapper(model)
        #         writer.add_graph(model_wrapper, x1)

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


def train_pcrlv2_3d(args, data_loader, run_dir, out_channel=3, writer=None):
    train_3d(args, data_loader, run_dir, out_channel=out_channel, writer=writer)


def train_cluster_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer, colors):

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    mg_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    end = time.time()
    for idx, (input1, input2, gt1, gt2, crop1_coords, crop2_coords, local_views) in enumerate(train_loader):

        B, C, H, W, D = input1.shape

        data_time.update(time.time() - end)

        x1 = input1.float()  # Crop 1
        x2 = input2.float()  # Crop 2

        if not args.cpu:
            x1 = x1.cuda()
            x2 = x2.cuda()
            gt1 = gt1.cuda()
            gt2 = gt2.cuda()
            crop1_coords = crop1_coords.cuda()
            crop2_coords = crop2_coords.cuda()

        # Get embeddings and predictions
        emb1, pred1 = model(x1)
        emb2, pred2 = model(x2)
        
        ## SwAV Loss
        
        # Normalize D dimension (BxNxD)
        emb1 = nn.functional.normalize(emb1, dim=2, p=2)  
        emb2 = nn.functional.normalize(emb2, dim=2, p=2) 

        # Get ground truths (teacher predictions)
        with torch.no_grad():
            # Get prototypes and normalize
            proto = model.module.prototypes.weight.data.clone()
            proto = nn.functional.normalize(proto, dim=1, p=2)  # Normalize D dimension (KxD)

            # Embedding to prototype similarity matrix
            cos_sim1 = torch.matmul(emb1, proto.t())  # BxNxK
            cos_sim2 = torch.matmul(emb2, proto.t())

            N = model.module.patch_num
            PH, PW, PD = model.module.patch_dim
            K = model.module.proto_num

            # Flatten batch and patch num dimensions of similarity matrices (BxNxK -> B*NxK), and transpose matrices (KxB*N) for sinkhorn algorithm
            flat_cos_sim1 = cos_sim1.reshape(B*N,K).T
            flat_cos_sim2 = cos_sim2.reshape(B*N,K).T

            # Standardize for numerical stability (maybe?)
            eps = 0.05
            flat_cos_sim1 = torch.exp(flat_cos_sim1 / eps)
            flat_cos_sim2 = torch.exp(flat_cos_sim2 / eps)

            # Teacher cluster assignments
            gt1 = sinkhorn(args, Q=flat_cos_sim1, nmb_iters=3).T.reshape((B,N,K))  # Also restore patch num dimension
            gt2 = sinkhorn(args, Q=flat_cos_sim2, nmb_iters=3).T.reshape((B,N,K))

            # Apply temperature
            temp = 1
            gt1 = gt1 / temp
            gt2 = gt2 / temp

        # Convert to probabilities
        # prob_gt1 = gt1.softmax(2)  # TODO: Have to check if this is already a probability
        # prob_gt2 = gt2.softmax(2)
        pred1 = pred1.softmax(2)
        pred2 = pred2.softmax(2)

        # Convert prediction and ground truth to (soft) cluster masks (restore spatial position of pooled image)
        NPH = H//PH  # Num patches at X
        NPW = W//PW  # Num patches at Y
        NPD = D//PD  # Num patches at Z
        pred1 = pred1.permute(0,2,1).reshape((B,K,NPH,NPW,NPD))
        pred2 = pred2.permute(0,2,1).reshape((B,K,NPH,NPW,NPD))
        gt1 = gt1.permute(0,2,1).reshape((B,K,NPH,NPW,NPD))
        gt2 = gt2.permute(0,2,1).reshape((B,K,NPH,NPW,NPD))

        # ROI-align crop intersection with cluster assignment intersection
        pred1, pred2, gt1, gt2 = roi_align_intersect(pred1, pred2, gt1, gt2, crop1_coords, crop2_coords)

        # Loss
        loss1 = swav_loss(gt1, gt2, pred1, pred2)

        # TODO: add the other losses later
        loss2 = torch.tensor(0)
        loss4 = 0
        local_loss = 0

        # Plot predictions on tensorboard
        with torch.no_grad():
            b_idx = 0
            if args.vis and idx==b_idx and epoch % 10 == 0:

                # Select 2D images
                img_idx = 0
                m_idx = 0
                c_idx = 0
                in1 = x1[img_idx,m_idx,:,:,input1.size(-1)//2].unsqueeze(0)
                in2 = x2[img_idx,m_idx,:,:,input2.size(-1)//2].unsqueeze(0)
                pred1 = pred1[img_idx,:,:,:,pred1.size(-1)//2].argmax(dim=0).unsqueeze(0)  # Take only hard cluster assignment (argmax)
                pred2 = pred2[img_idx,:,:,:,pred2.size(-1)//2].argmax(dim=0).unsqueeze(0)
                gt1 = gt1[img_idx,:,:,:,gt1.size(-1)//2].argmax(dim=0).unsqueeze(0)
                gt2 = gt2[img_idx,:,:,:,gt2.size(-1)//2].argmax(dim=0).unsqueeze(0)

                # Min-max norm input images
                in1 = (in1 - in1.min())/(in1.max() - in1.min())
                in2 = (in2 - in2.min())/(in2.max() - in2.min())

                # Interpolate cluster masks to original input shape
                pred1 = f.interpolate(pred1.float().unsqueeze(0), size=(H,W)).squeeze(0)
                pred2 = f.interpolate(pred2.float().unsqueeze(0), size=(H,W)).squeeze(0)
                gt1 = f.interpolate(gt1.float().unsqueeze(0), size=(H,W)).squeeze(0)
                gt2 = f.interpolate(gt2.float().unsqueeze(0), size=(H,W)).squeeze(0)

                # Send to cpu
                in1 = in1.cpu().detach()
                in2 = in2.cpu().detach()
                pred1 = pred1.cpu().detach()
                pred2 = pred2.cpu().detach()
                gt1 = gt1.cpu().detach()
                gt2 = gt2.cpu().detach()

                # Give color to each cluster in cluster masks
                pred1 = pred1.repeat((3,1,1)).permute(1,2,0)  # Convert to RGB and move channel dim to the end
                pred2 = pred2.repeat((3,1,1)).permute(1,2,0)
                gt1 = gt1.repeat((3,1,1)).permute(1,2,0)
                gt2 = gt2.repeat((3,1,1)).permute(1,2,0)
                for i in range(colors.shape[0]):
                    pred1[pred1[:,:,0] == i] = colors[i]
                    pred2[pred2[:,:,0] == i] = colors[i]
                    gt1[gt1[:,:,0] == i] = colors[i]
                    gt2[gt2[:,:,0] == i] = colors[i]
                pred1 = pred1.permute(2,1,0)
                pred2 = pred2.permute(2,1,0)
                gt1 = gt1.permute(2,1,0)
                gt2 = gt2.permute(2,1,0)

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

                in_img_name = 'img/train/raw' # f'b{b_idx}_img{img_idx}_slc{slc_idx}_raw'
                pred_img_name = 'img/train/pred' # f'b{b_idx}_img{img_idx}_slc{slc_idx}_pred'
                gt_img_name = 'img/train/gt' # f'b{b_idx}_img{img_idx}_slc{slc_idx}_gt'

                writer.add_image(in_img_name, img_tensor=in_img, global_step=epoch, dataformats='CHW')
                writer.add_image(pred_img_name, img_tensor=pred_img, global_step=epoch, dataformats='CHW')   
                writer.add_image(gt_img_name, img_tensor=gt_img, global_step=epoch, dataformats='CHW')

        # Total Loss
        # TODO: add the other losses later
        loss = loss1 

        # Backward
        if loss > 1000 and epoch > 10:
            print('skip the step')
            continue
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
        total_loss_meter.update(loss, B)
        if not args.cpu:
            torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # if args.tensorboard:
        #     if epoch == 0:  # Only on the first iteration, write model graph on tensorboard
        #         model_wrapper = TraceWrapper(model)
        #         writer.add_graph(model_wrapper, x1)

        # print info
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


def val_cluster_inner(args, epoch, val_loader, model, colors):

    with torch.no_grad():

        grid_preds = []  # For plotting a grid epoch X image: A row of predicted cluster assignments for each image in the current epoch

        model.eval()

        for idx, (image, _, gt, _, crop_coords, _, _) in enumerate(val_loader):
            
            if idx == 1:
                break  # Stop after the first batch

            B, _, H, W, D = image.shape
            N = model.module.patch_num
            PH, PW, PD = model.module.patch_dim
            K = model.module.proto_num

            # Keep only modality 0
            image = image[:,0:1,:,:,:]

            x = image.float()

            if not args.cpu:
                x = x.cuda()
                gt = gt.cuda()

            # Get embeddings and predictions
            _, pred = model(x)

            # Convert to probabilities
            pred = pred.softmax(2)

            # Convert prediction to (soft) cluster masks (restore spatial position of pooled image)
            NPH = H//PH  # Num patches at X
            NPW = W//PW  # Num patches at Y
            NPD = D//PD  # Num patches at Z
            pred = pred.permute(0,2,1).reshape((B,K,NPH,NPW,NPD))

            # Plot prediction grid on tensorboard
            n_epochs = min(10,args.epochs+1) # The number of epochs to sample from for the grid (10 or all epochs if total less than 10) (+1 because we always run for one extra epoch)
            n_images = min(10,args.b)  # The number of images to sample from for the grid (10 or all images if total less than 10)
            if args.vis and epoch % (args.epochs + 1 // n_epochs) == 0: 
               
                # If epoch 0, then also add the input images as the first row of the grid
                if epoch == 0:  
                    for img_idx in range(n_images):
                        x_i = x[img_idx,0,:,:,D//2]                   
                        x_i = (x_i - x_i.min())/(x_i.max() - x_i.min())  # Min-max norm input images
                        x_i = x_i.repeat((3,1,1)).permute(1,2,0)  # Convert to RGB and move channel dim to the end
                        x_i = x_i.cpu().detach()
                        grid_preds.append(x_i)
                
                 # Next, add the predictions for each image at the current epoch
                for img_idx in range(n_images): 
                    pred_i = pred[img_idx,:,:,:,NPD//2].argmax(dim=0).unsqueeze(0)  # Take only hard cluster assignment (argmax)
                    pred_i = f.interpolate(pred_i.float().unsqueeze(0), size=(H,W)).squeeze(0)  # Interpolate cluster masks to original input shape
                    pred_i = pred_i.repeat((3,1,1)).permute(1,2,0)  # Convert to RGB and move channel dim to the end
                    for i in range(colors.shape[0]):  # Give color to each cluster in cluster masks
                        pred_i[pred_i[:,:,0] == i] = colors[i]
                    pred_i = pred_i.cpu().detach()
                    grid_preds.append(pred_i)

    return grid_preds


def train_cluster_3d(args, data_loader, run_dir, out_channel=3, writer=None):
    train_3d(args, data_loader, run_dir, out_channel=out_channel, writer=writer)