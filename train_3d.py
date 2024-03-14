"""
Training code for C2L

"""
from __future__ import print_function

import os
import sys
import time
import math
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as f
import random
from utils import adjust_learning_rate, AverageMeter, sinkhorn, swav_loss, roi_align
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
from matplotlib import pyplot as plt

from models import PCRLv23d, TraceWrapper

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


def train_pcrlv2_3d(args, data_loader, out_channel=3):

    # Generate colors for cluster masks (make 10 of them, but later TODO: base it on cluster number K)
    palette = sns.color_palette()
    colors = torch.Tensor([list(color) for color in palette])

    torch.backends.cudnn.deterministic = True

    curr_time = str(time.time()).replace(".","")
    run_name = f'{args.model}{"_sc" if args.skip_conn else ""}_{args.n}_{args.phase}_b{args.b}_e{args.epochs}_lr{"{:f}".format(args.lr).split(".")[-1]}_t{curr_time}'

    writer = None
    if args.tensorboard:
        # Create tensorboard writer
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        writer = SummaryWriter(os.path.join(args.output, run_name))

    train_loader = data_loader['train']

    # Create model and optimizer
    model = PCRLv23d(skip_conn=args.skip_conn)
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
    if not args.cpu:
        criterion = criterion.cuda()

    cosine = nn.CosineSimilarity()
    if not args.cpu:
        cosine = cosine.cuda()
        cudnn.benchmark = True

    for epoch in range(0, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()

        loss, prob, total_loss, writer = train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer, colors)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if args.tensorboard:
            writer.add_scalar('loss/train', total_loss, epoch)  # Write train loss on tensorboard

        # Save model
        if epoch % 100 == 0 or epoch == 240:
            print('==> Saving...')
            state = {'opt': args, 'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}

            save_file = os.path.join(args.output, run_name + '.pt')
            torch.save(state, save_file)

            # Help release GPU memory
            del state
        if not args.cpu:
            torch.cuda.empty_cache()


def cos_loss(cosine, output1, output2):
    index = random.randint(0, len(output1) - 1)  # Because we select a feature map from a random scale
    sample1 = output1[index]
    sample2 = output2[index]
    loss = -(cosine(sample1[1], sample2[0].detach()).mean() + cosine(sample2[1],
                                                                     sample1[0].detach()).mean()) * 0.5
    return loss, index


def train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer, colors):
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
    for idx, (input1, input2, gt, gt2, crop1_coords, crop2_coords, local_views) in enumerate(train_loader):

        B, C, H, W, D = input1.shape

        data_time.update(time.time() - end)

        bsz = input1.size(0)
        x1 = input1.float()
        x2 = input2.float()

        if not args.cpu:
            x1 = x1.cuda()
            x2 = x2.cuda()
            gt = gt.cuda()

        mask1, decoder_outputs1, middle_masks1, encoder_output1 = model(x1)

        if args.model == 'pcrlv2':
            mask2, decoder_outputs2, _, encoder_output2 = model(x2)
            loss2, index2 = cos_loss(cosine, decoder_outputs1, decoder_outputs2)
            local_loss = 0.0
            local_input = torch.cat(local_views, dim=0)  # 6 * bsz, 3, d, 96, 96
            _, local_views_outputs, _, _ = model(local_input, local=True)  # 4 * 2 * [6 * bsz, 3, d, 96, 96]
            local_views_outputs = [torch.stack(t) for t in local_views_outputs]
            for i in range(len(local_views)):
                local_views_outputs_tmp = [t[:, bsz * i: bsz * (i + 1)] for t in local_views_outputs]
                loss_local_1, _ = cos_loss(cosine, decoder_outputs1, local_views_outputs_tmp)
                loss_local_2, _ = cos_loss(cosine, decoder_outputs2, local_views_outputs_tmp)
                local_loss += loss_local_1
                local_loss += loss_local_2
            local_loss = local_loss / (2 * len(local_views))
            loss1 = criterion(mask1, gt)
            beta = 0.5 * (1. + math.cos(math.pi * epoch / 240))
            loss4 = beta * criterion(middle_masks1[index2], gt)  
            loss = loss1 + loss2 + loss4 + local_loss

        elif args.model == 'cluster':
            # x2 = gt  #TODO: in the future make the encoder_output2 come from the second crop
            _, _, _, encoder_output2 = model(x2) 

            ## SwAV Loss

            # Flatten spatial dims of feature map
            B, N, PH, PW, PD = encoder_output1.shape  # Batch, Number of patches, Patch spatial dims
            encoder_output1 = encoder_output1.reshape(B, N, PH*PW*PD)
            encoder_output2 = encoder_output2.reshape(B, N, PH*PW*PD)

            # Get embeddings and outputs and normalize embeddings
            emb1, out1 = model.module.forward_cluster_head(encoder_output1)
            emb2, out2 = model.module.forward_cluster_head(encoder_output2)
            
            # Normalize D dimension (BxNxD)
            emb1 = nn.functional.normalize(emb1, dim=2, p=2)  
            emb2 = nn.functional.normalize(emb2, dim=2, p=2) 

            # Get ground truths
            with torch.no_grad():
                # Get prototypes and normalize
                proto = model.module.prototypes.weight.data.clone()
                proto = nn.functional.normalize(proto, dim=1, p=2)  # Normalize D dimension (KxD)

                # Embedding to prototype similarity matrix
                cos_sim1 = torch.matmul(emb1, proto.t())  # BxNxK
                cos_sim2 = torch.matmul(emb2, proto.t())

                B, N, K = cos_sim1.shape

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
            prob1 = out1.softmax(2)
            prob2 = out2.softmax(2)

            # Convert prediction and ground truth to cluster masks (restore spatial position of patched image)
            NPH = H//PH  # Number of patches at X dimension
            NPW = W//PW  # Number of patches at Y dimension
            NPD = D//PD  # Number of patches at Z dimension
            pred1 = prob1.argmax(dim=2).reshape((B,1,NPH,NPW,NPD))  # TODO: Later, remove argmax and use soft clusters
            pred2 = prob2.argmax(dim=2).reshape((B,1,NPH,NPW,NPD))
            gt1 = gt1.argmax(dim=2).reshape((B,1,NPH,NPW,NPD))
            gt2 = gt2.argmax(dim=2).reshape((B,1,NPH,NPW,NPD))

            # ROI-align
            pred1, pred2 = roi_align(pred1, pred2, crop1_coords, crop2_coords)

            # Loss
            loss1 = swav_loss(gt1, gt2, prob1, prob2)  # TODO: In the future, use encoder_output_2 and roi align the clusters
            # TODO: add the other losses later
            loss2 = torch.tensor(0)
            loss4 = 0
            local_loss = 0

            # Plot predictions on tensorboard
            b_idx = 0
            if args.vis and idx==b_idx and epoch % 10 == 0:

                # Select 2D images
                img_idx = 0
                m_idx = 0
                c_idx = 0
                in1 = x1[img_idx,m_idx,:,:,input1.size(-1)//2].unsqueeze(0)
                in2 = x2[img_idx,m_idx,:,:,input2.size(-1)//2].unsqueeze(0)
                pred1 = pred1[img_idx,:,:,:,pred1.size(-1)//2]
                pred2 = pred2[img_idx,:,:,:,pred2.size(-1)//2]
                gt1 = gt1[img_idx,:,:,:,gt1.size(-1)//2]
                gt2 = gt2[img_idx,:,:,:,gt2.size(-1)//2]

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
                pred1 = pred1.repeat((3,1,1)).permute(1,2,0)
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

                in_img_name = 'img/raw' # f'b{b_idx}_img{img_idx}_slc{slc_idx}_raw'
                pred_img_name = 'img/pred' # f'b{b_idx}_img{img_idx}_slc{slc_idx}_pred'
                gt_img_name = 'img/gt' # f'b{b_idx}_img{img_idx}_slc{slc_idx}_gt'

                writer.add_image(in_img_name, img_tensor=in_img, global_step=epoch, dataformats='CHW')
                writer.add_image(pred_img_name, img_tensor=pred_img, global_step=epoch, dataformats='CHW')   
                writer.add_image(gt_img_name, img_tensor=gt_img, global_step=epoch, dataformats='CHW')

        # Total Loss
        if args.model == 'pcrlv2':
            loss = loss1 + loss2 + loss4 + local_loss
        if args.model == 'cluster':  # TODO: add the other losses later
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

        if args.tensorboard:
            if epoch == 0:  # Only on the first iteration, write model graph on tensorboard
                model_wrapper = TraceWrapper(model)
                writer.add_graph(model_wrapper, x1)

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
