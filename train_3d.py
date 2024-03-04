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
import random
from utils import adjust_learning_rate, AverageMeter, sinkhorn
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import roi_align

from matplotlib import pyplot as plt


from models import PCRLv23d, TraceWrapper

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


def train_pcrlv2_3d(args, data_loader, out_channel=3):

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
    # create model and optimizer
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

        loss, prob, total_loss, writer = train_pcrlv2_inner(args, epoch, train_loader, model, optimizer, criterion, cosine, writer)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if args.tensorboard:
            writer.add_scalar('loss/train', total_loss, epoch)  # Write train loss on tensorboard

        # save model
        if epoch % 100 == 0 or epoch == 240:
            # saving the model
            print('==> Saving...')
            state = {'opt': args, 'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(), 'epoch': epoch}

            save_file = os.path.join(args.output, run_name + '.pt')
            torch.save(state, save_file)

            # help release GPU memory
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
    for idx, (input1, input2, gt, gt2, local_views) in enumerate(train_loader):

        data_time.update(time.time() - end)

        bsz = input1.size(0)
        x1 = input1.float()
        x2 = input2.float()

        if not args.cpu:
            x1 = x1.cuda()
            x2 = x2.cuda()
            gt = gt.cuda()

        mask1, decoder_outputs1, middle_masks1, encoder_output1 = model(x1)

        if args.phase == 'pretask':
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

        elif args.phase == 'cluster_pretask':
            _, _, _, encoder_output2 = model(gt)  #TODO: in the future make the encoder_output2 come from the second crop

            ## SwAV Algorithm

            # Flatten spatial dims of feature map
            # B, N, P1, P2, P3 = encoder_output1.shape[2:] TODO: save the spatial dims and give the following P1,P2 different names
            P1 = torch.prod(torch.tensor(encoder_output1.shape[2:], dtype=int)).item()
            P2 = torch.prod(torch.tensor(encoder_output2.shape[2:], dtype=int)).item()
            encoder_output1 = encoder_output1.reshape(encoder_output1.shape[0], encoder_output1.shape[1], P1)
            encoder_output2 = encoder_output2.reshape(encoder_output2.shape[0], encoder_output2.shape[1], P2)

            # Get embeddings and outputs and normalize embeddings
            emb1, out1 = model.module.forward_cluster_head(encoder_output1)
            emb2, out2 = model.module.forward_cluster_head(encoder_output2)
            emb1 = nn.functional.normalize(emb1, dim=2, p=2)  # Normalize D dimension (BxNxD)
            emb2 = nn.functional.normalize(emb2, dim=2, p=2)  # Normalize D dimension (BxNxD)

            # Get prototypes and normalize
            with torch.no_grad():
                proto = model.module.prototypes.weight.data.clone()
                proto = nn.functional.normalize(proto, dim=1, p=2)  # Normalize D dimension (KxD)

            # Embedding to prototype similarity matrix
            cos_sim1 = torch.matmul(emb1, proto.t())  # BxNxK
            cos_sim2 = torch.matmul(emb2, proto.t())

            B, N, K = cos_sim1.shape

            # Flatten batch and patch dimensions of similarity matrices (BxNxK -> B*NxK), and transpose matrices (KxB*N) for sinkhorn algorithm
            flat_cos_sim1 = cos_sim1.reshape(K, B*N)
            flat_cos_sim2 = cos_sim2.reshape(K, B*N)

            # Standardize for numerical stability (maybe?)
            eps = 0.05
            flat_cos_sim1 = torch.exp(flat_cos_sim1 / eps)
            flat_cos_sim2 = torch.exp(flat_cos_sim2 / eps)

            with torch.no_grad():
                # Teacher cluster assignments
                gt1 = sinkhorn(args, Q=flat_cos_sim1, nmb_iters=3).reshape((B,N,K))
                gt2 = sinkhorn(args, Q=flat_cos_sim2, nmb_iters=3).reshape((B,N,K))

                # Apply temperature
                temp = 1
                gt1 = gt1 / temp
                gt2 = gt2 / temp

            # Convert to probabilities
            prob_gt1 = gt1.softmax(2)
            prob_gt2 = gt2.softmax(2)
            prob1 = out1.softmax(2)
            prob2 = out2.softmax(2)

            # Plot cluster assignments

            # cluster1 = prob1.reshape((B,int(torch.sqrt(N)),int(torch.))).argmax(dim=3)  # TODO: define cluster1 and cluster2
            # TODO: implement cluster output saving for monitoring
            # if idx == 0:
            #    img_id = 0
            #    slice_id = 0
            #    plt.imshow(x1[img_id,:,:,slice_id].detach().cpu().numpy())
            #    plt.save(os.path.join(writer_log_dir, f'b{idx}_img{img_id}_e{epoch}_raw.png'))

            #    cluster1 = torch.argmax(prob1, axis=2).reshape((B, int(N / 2), int(N / 2)))
            #    plt.imshow(cluster1[img_id,:,:,slice_id].detach().cpu().numpy())
            #    plt.save(os.path.join(writer_log_dir, f'b{idx}_img{img_id}_e{epoch}_cluster1.png'))

            #    cluster2 = torch.argmax(prob2, axis=2).reshape((B, int(N / 2), int(N / 2)))
            #    plt.imshow(cluster2[img_id,:,:,slice_id].detach().cpu().numpy())
            #    plt.save(os.path.join(writer_log_dir, f'b{idx}_img{img_id}_e{epoch}_cluster2.png'))

            loss1 = swav_loss(prob_gt1, prob_gt2, prob1, prob2)  # TODO: In the future, use encoder_output_2 and roi align the clusters
            # TODO: add the other losses later
            loss2 = torch.tensor(0)
            loss4 = 0
            local_loss = 0

        if args.phase == 'pretask':
            loss = loss1 + loss2 + loss4 + local_loss
        if args.phase == 'cluster_pretask':  # TODO: add the other losses later
            loss = loss1 

        # ===================backward=====================
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

        # ===================meters=====================
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

def swav_loss(gt1, gt2, out1, out2):
    loss = - 0.5 * torch.mean(gt1 * torch.log(out2) + gt2 * torch.log(out1))
    return loss
