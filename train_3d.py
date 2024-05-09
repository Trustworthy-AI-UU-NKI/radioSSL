"""
Training code for C2L

"""
from __future__ import print_function

import sys
import time
import math
import random
import PIL
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms.functional as t
from torch import autocast
from torch import autocast
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler
else:
    from torch.cpu.amp import GradScaler

from models import PCRLv23d, Cluster3d, TraceWrapper
from tools import adjust_learning_rate, AverageMeter, sinkhorn, swav_loss, roi_align_intersect


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


def train_3d(args, data_loader, run_dir, writer=None):

    if 'cluster' in args.model:
        # Generate colors for cluster masks
        palette = sns.color_palette(palette='bright', n_colors=args.k)
        colors = torch.Tensor([list(color) for color in palette]).cpu()  # cpu because we apply it on a detached tensor later

    torch.backends.cudnn.deterministic = True

    train_loader = data_loader['train']
    val_loader = data_loader['eval']

    # Create model and optimizer
    if args.model == 'pcrlv2':
        model = PCRLv23d(skip_conn=args.skip_conn)
    elif 'cluster' in args.model:
        model = Cluster3d(n_clusters=args.k, seed=args.seed)
    if not args.cpu:
        model = model.cuda()

    scaler = GradScaler()
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
        cudnn.benchmark = True

    grid_pred_all = []  # Clustering task predictions to visualize in a grid (epoch X image) for each scale

    for epoch in range(0, args.epochs + 1):

        # TRAINING

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        if args.model == 'pcrlv2':
            loss, prob, total_loss, writer = train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, scaler, criterion, cosine, writer)
        elif 'cluster' in args.model:
            loss, prob, total_loss, writer = train_cluster_inner(args, epoch, train_loader, model, optimizer, scaler, writer, colors)

        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if args.tensorboard:
            writer.add_scalar('loss/train', total_loss, epoch)  # Write train loss on tensorboard


        # VALIDATION (only for clustering task, just for visualization purposes)

        # n_epochs = min(10,args.epochs+1) # The number of epochs to sample for the grid (10 or all epochs if total less than 10) (+1 because we always run for one extra epoch)
        
        # if args.vis and ((epoch % ((args.epochs + 1) // (n_epochs - 2))) == 0 or epoch==args.epochs) and 'cluster' in args.model and args.n == 'brats': 
        #     # TODO: Currently only works for BraTS Clustering
        #     # The n_epochs - 2 is because we want to sample one less so that we can always add the final epoch in the end regardless of the step, and one less because epoch 0 is always included

        #     print("==> validating...")
            
        #     # Validate
        #     row_pred_all = val_cluster_inner(args, epoch, val_loader, model, colors)  # Array of a row for each scale to add to the grid of each scale
            
        #     # Add row to corresponding grid for each scale
        #     if len(grid_pred_all) == 0:
        #         grid_pred_all = row_pred_all
        #     else:
        #         for i in range(len(row_pred_all)):
        #             grid_pred_all[i].extend(row_pred_all[i])
            
        #     n_scales = len(grid_pred_all)
        #     n_cols = min(10,args.b)
        #     n_rows = len(grid_pred_all[0]) // n_cols
            
        #     # Plot for every scale its grid of predictions for sampled epochs up to now
        #     for sc in range(n_scales):
        #         fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15*(n_rows/n_cols)))
        #         for i, ax in enumerate(axes.flat):
        #             ax.imshow(grid_pred_all[sc][i]) 
        #             ax.axis('off')  # Turn off axis labels
        #             if i % n_cols:
        #                 ax.set_ylabel(f'Epoch {epoch}', rotation=0, size='large')
        #         plt.tight_layout()  # Adjust spacing between subplots
        #         # Save grid to buffer and then log on tensorboard
        #         buf = io.BytesIO()
        #         plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        #         buf.seek(0)
        #         grid = PIL.Image.open(buf)
        #         grid = t.pil_to_tensor(grid)
        #         writer.add_image(f'img/val/grid_{sc}', img_tensor=grid, global_step=epoch)

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

        
def train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, scaler, criterion, cosine, writer):
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
        scaler.scale(loss).backward()
        scaler.step(optimizer)

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


def train_pcrlv2_3d(args, data_loader, run_dir, writer=None):
    train_3d(args, data_loader, run_dir, writer=writer)


def train_cluster_inner(args, epoch, train_loader, model, optimizer, scaler, writer, colors):

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    mg_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    total_loss_meter = AverageMeter()

    end = time.time()
    for idx, (input1, input2, _, _, crop1_coords, crop2_coords, _) in enumerate(train_loader):

        B, C, H, W, D = input1.shape

        data_time.update(time.time() - end)

        x1 = input1.float()  # Crop 1
        x2 = input2.float()  # Crop 2

        if not args.cpu:
            x1 = x1.cuda()
            x2 = x2.cuda()
            crop1_coords = crop1_coords.cuda()
            crop2_coords = crop2_coords.cuda()

        device_type = 'cpu' if args.cpu else 'cuda'
        with autocast(device_type=device_type):  # Run in mixed-precision

            # STUDENT CLUSTER ASSIGNMENT ------------------------------------

            # Get cluster predictions from student U-Net
            pred1 = model.module(x1)
            pred2 = model.module(x2)
            
            # Convert to probabilities
            pred1 = pred1.softmax(2)
            pred2 = pred2.softmax(2)

            # TEACHER CLUSTER ASSIGNMENT ------------------------------------

            with torch.no_grad():
                # Get upsampled features from teacher DINO ViT16 encoder and flatten spatial dimensions to get feature vectors for each pixel
                feat_vec1 = []
                feat_vec2 = []
                for i in range(B):  # For cuda memory efficiency, enter only 1 batch at a time (we have a lot)
                    #  B x 1 x H x W x D -(Flatten slices)-> B*D x 1 x H x W -(RGB)->  B*D x 3 x H x W -(FeatUp)-> B*D x C' x H x W -(Vectorize)-> B*D*H*W x C' 
                    feat_vec1 = model.module.featup_upsampler(x1[i].unsqueeze(0).repeat(1,3,1,1)).permute(0,2,3,1).flatten(0,2)  
                    feat_vec2 = model.module.featup_upsampler(x2[i].unsqueeze(0).repeat(1,3,1,1)).permute(0,2,3,1).flatten(0,2) 
                feat_vec1 = torch.cat(feat_vec1)
                
                # Perform K-Means on teacher feature vectors
                K = model.module.kmeans.n_clusters
                # gt_vec1 = model.module.kmeans.fit_predict(x=feat_vec1.unsqueeze(0))
                # gt_vec2 = model.module.kmeans.fit_predict(x=feat_vec2.unsqueeze(0))
                model.module.kmeans = model.module.kmeans.fit(torch.cat([feat_vec1,feat_vec2]).detach().cpu().numpy())
                gt_vec = torch.from_numpy(model.module.kmeans.predict(torch.cat([feat_vec1,feat_vec2]).detach().cpu().numpy())).cuda().to(torch.int64)

                if not args.cpu:
                    gt_vec1 = gt_vec1.cuda()
                    gt_vec2 = gt_vec2.cuda()
        
                # Convert to one-hot encoding and restore spatial dimensions
                gt1 = f.one_hot(gt_vec[:gt_vec.shape[0]//2], K).reshape(-1, H, W, K).permute(0,4,2,3,1)  # B*D*H*W x K -> B x D x H x W x K -> B x K x H x W x D
                gt2 = f.one_hot(gt_vec[gt_vec.shape[0]//2:], K).reshape(-1, H, W, K).permute(0,4,2,3,1)

            # --------------------------------------------------------------

        # ROI-align crop intersection with cluster assignment intersection
        roi_pred1, roi_pred2, roi_gt1, roi_gt2 = roi_align_intersect(pred1, pred2, gt1, gt2, crop1_coords, crop2_coords)

        # SwAV Loss for current scale
        scale_swav_loss = swav_loss(roi_gt1, roi_gt2, roi_pred1, roi_pred2)

        # Plot predictions on tensorboard
        with torch.no_grad():
            b_idx = 0
            if args.vis and idx==b_idx: #and epoch % 10 == 0:

                # Select 2D images
                img_idx = 0
                m_idx = 0
                s_idx = D//2
                in1 = x1[img_idx,m_idx,:,:,s_idx].unsqueeze(0)
                in2 = x2[img_idx,m_idx,:,:,s_idx].unsqueeze(0)
                pred1 = pred1[img_idx,:,:,:,s_idx].argmax(dim=0).unsqueeze(0)  # Take only hard cluster assignment (argmax)
                pred2 = pred2[img_idx,:,:,:,s_idx].argmax(dim=0).unsqueeze(0)
                gt1 = gt1[img_idx,:,:,:,s_idx].argmax(dim=0).unsqueeze(0)
                gt2 = gt2[img_idx,:,:,:,s_idx].argmax(dim=0).unsqueeze(0)

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
                for c in range(colors.shape[0]):
                    pred1[pred1[:,:,0] == c] = colors[c]
                    pred2[pred2[:,:,0] == c] = colors[c]
                    gt1[gt1[:,:,0] == c] = colors[c]
                    gt2[gt2[:,:,0] == c] = colors[c]
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

                in_img_name = 'img/train/raw' 
                pred_img_name = f'img/train/pred'
                gt_img_name = f'img/train/gt'

                writer.add_image(in_img_name, img_tensor=in_img, global_step=epoch, dataformats='CHW')
                writer.add_image(pred_img_name, img_tensor=pred_img, global_step=epoch, dataformats='CHW')   
                writer.add_image(gt_img_name, img_tensor=gt_img, global_step=epoch, dataformats='CHW')

        # TODO: add the other losses later
        loss1 = scale_swav_loss
        loss2 = torch.tensor(0)
        loss4 = 0
        local_loss = 0

        # Total Loss
        # TODO: add the other losses later
        loss = loss1 

        # Backward
        if loss > 1000 and epoch > 10:
            print('skip the step')
            continue
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)

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

        # This array is for plotting a grid epoch X image for each scale:
        # Each row contains the predicted cluster assignments of a specific scale for each image in the current epoch
        grid_pred_all = []  

        model.eval()

        for idx, (image, _, gt, _, crop_coords, _, _) in enumerate(val_loader):
            
            if idx != 0:
                continue  # Validate only batch 2

            B, _, H, W, D = image.shape
            K = model.module.proto_num

            # Keep only modality 0
            image = image[:,0:1,:,:,:]

            x = image.float()

            if not args.cpu:
                x = x.cuda()
                gt = gt.cuda()

            # Get embeddings and predictions
            _, pred_all = model(x)

            n_scales = len(pred_all)
            for sc in range(n_scales):  # For each scale

                grid_pred = []  # Contains the grid predictions of a specific scale

                N = model.module.patch_num[sc]
                PH, PW, PD = model.module.patch_dim[sc]

                pred = pred_all[sc]

                # Convert to probabilities
                pred = pred.softmax(2)

                # Convert prediction to (soft) cluster masks (restore spatial position of pooled image)
                NPH = H//PH  # Num patches at X
                NPW = W//PW  # Num patches at Y
                NPD = D//PD  # Num patches at Z
                pred = pred.permute(0,2,1).reshape((B,K,NPH,NPW,NPD))

                # Gather predictions to visualize on grid
                n_images = min(10,args.b)  # The number of images to sample from for the grid (10 or all images if total less than 10)
                if epoch == 0:  # If first epoch, add the input images as the first row of the grid
                    for img_idx in range(n_images):
                        x_i = x[img_idx,0,:,:,D//2]                   
                        x_i = (x_i - x_i.min())/(x_i.max() - x_i.min())  # Min-max norm input images
                        x_i = x_i.repeat((3,1,1)).permute(1,2,0)  # Convert to RGB and move channel dim to the end
                        x_i = x_i.cpu().detach()
                        grid_pred.append(x_i)
                for img_idx in range(n_images): # Next, add the predictions for each image at the current epoch as the next row
                    pred_i = pred[img_idx,:,:,:,NPD//2].argmax(dim=0).unsqueeze(0)  # Take only hard cluster assignment (argmax)
                    pred_i = f.interpolate(pred_i.float().unsqueeze(0), size=(H,W)).squeeze(0)  # Interpolate cluster masks to original input shape
                    pred_i = pred_i.repeat((3,1,1)).permute(1,2,0)  # Convert to RGB and move channel dim to the end
                    pred_i = pred_i.cpu().detach() # Send pred and color tensors to cpu
                    colors = colors.cpu()
                    for c in range(colors.shape[0]):  # Give color to each cluster in cluster masks
                        pred_i[pred_i[:,:,0] == c] = colors[c]
                    pred_i = pred_i.cpu().detach()
                    grid_pred.append(pred_i)
                
                grid_pred_all.append(grid_pred)

    return grid_pred_all


def train_cluster_3d(args, data_loader, run_dir, writer=None):
    train_3d(args, data_loader, run_dir, writer=writer)