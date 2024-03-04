import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import random

from utils import prepare_model, get_loss, dice_coeff
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt


def train_chest_classification(args, dataloader):
    criterion = nn.BCELoss()
    train_generator = dataloader['train']
    valid_generator = dataloader['eval']
    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=0.2,  # dropout ratio, default is None
        activation='sigmoid',  # activation function, default is None
        classes=14,  # define number of output labels
    )
    model = smp.Unet('resnet18', in_channels=3, aux_params=aux_params, classes=1, encoder_weights=None)
    if args.weight is not None:
        weight_path = args.weight
        if args.cpu:
            encoder_dict = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        else:
            encoder_dict = torch.load(weight_path)['state_dict']
        encoder_dict['fc.bias'] = 0
        encoder_dict['fc.weight'] = 0
        model.encoder.load_state_dict(encoder_dict)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())]).cuda()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 100000
    num_epoch_no_improvement = 0

    for epoch in range(0, args.epochs + 1):
        scheduler.step(epoch)
        model.train()
        for iteration, (image, gt) in enumerate(train_generator):
            image = image.cuda().float()
            gt = gt.cuda().float()
            _, pred = model(image)
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 4))
            if (iteration + 1) % 5 == 0:
                print('Epoch [{}/{}], iteration {}, Loss:{:.6f}, {:.6f}'
                      .format(epoch + 1, args.epochs, iteration + 1, loss.item(), np.average(train_losses)))
                sys.stdout.flush()
        with torch.no_grad():
            model.eval()
            print("validating....")
            for i, (x, y) in enumerate(valid_generator):
                x = x.cuda()
                y = y.cuda().float()
                _, pred = model(x)
                loss = criterion(pred, y)
                valid_losses.append(loss.item())

        # logging
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss,
                                                                                    train_loss))
        train_losses = []
        valid_losses = []
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),  # only save encoder
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.output,
                            args.model + "_" + args.n + '_' + args.phase + '_' + str(args.ratio) + '.pt'))
            print("Saving model ", os.path.join(args.output, args.model + "_" + args.n + '_' + args.phase + '_' + str(
                args.ratio) + '.pt'))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                      num_epoch_no_improvement))
            num_epoch_no_improvement += 1
            if num_epoch_no_improvement == args.patience:
                print("Early Stopping")
                break
        sys.stdout.flush()


def train_segmentation(args, dataloader, in_channels, n_classes, run_dir, writer=None):

    # Get data
    train_generator = dataloader['train']
    valid_generator = dataloader['eval']

    # Get model
    model = prepare_model(args, in_channels, n_classes)

    # Parallelize model
    if args.cpu:
        device_count = 0
    else:
        device_count = torch.cuda.device_count()
    model = nn.DataParallel(model, device_ids=[i for i in range(device_count)])
    if not args.cpu:
        model = model.cuda()

    # Loss
    criterion = get_loss(args.n)

    # Optimizer
    if args.model == 'genesis':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        
    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    avg_train_losses = []
    avg_valid_losses = []
    best_loss = 100000
    num_epoch_no_improvement = 0

    for epoch in range(0, args.epochs + 1):
        train_losses = []
        valid_losses = []
        scheduler.step(epoch)
        model.train()
        for iteration, (image, gt) in enumerate(train_generator):
            if not args.cpu:
                image = image.cuda()
                gt = gt.cuda()
            image = image.float()
            gt = gt.float()

            if args.d == 2: # If model is 2D unet, then combine batch and slice dimension and scale input to power of 2
                # Input dimensions
                B, M, _, S, _ = image.shape
                _, C, _, _, _ = gt.shape 
                H, W = (128,128)
                # Combine batch and slice dim
                image = image.permute(0,3,1,2,4).flatten(0,1)  # B x M x H x S x W -> B*S x M x H x W
                gt = gt.permute(0,3,1,2,4).flatten(0,1)
                # Scale
                image = f.interpolate(image, size=(H,W))
                gt = f.interpolate(gt, size=(H,W))

            pred = model(image)

            if args.d == 2: # If 2D unet, then revert to original dims
                image = image.reshape(B,S,M,H,W).permute(0,2,3,1,4)
                gt = gt.reshape(B,S,C,H,W).permute(0,2,3,1,4)
                pred = f.sigmoid(pred.reshape(B,S,C,H,W).permute(0,2,3,1,4))  # Also apply sigmoid becauce the 2D model doesn't

            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            if (iteration + 1) % 5 == 0:
                print('Epoch [{}/{}], iteration {}, Loss:{:.6f}, {:.6f}'
                      .format(epoch + 1, args.epochs, iteration + 1, loss.item(), np.average(train_losses)))
                sys.stdout.flush()

            b_idx = 0
            if args.vis and iteration==b_idx and epoch % 5 == 0:
                mod_idx = 0
                c_idx = 0
                img_idx = 0
                slc_idx = image.shape[3] // 2

                raw_img = image[b_idx,mod_idx,:,slc_idx,:].cpu().detach().numpy()
                gt_img = gt[b_idx,:,:,slc_idx,:].cpu().detach().numpy()
                pred_img = pred[b_idx,:,:,slc_idx,:].cpu().detach().numpy()

                raw_img_name = f'b{b_idx}_img{img_idx}_slc{slc_idx}_raw'
                gt_img_name = f'b{b_idx}_img{img_idx}_slc{slc_idx}_gt'
                pred_img_name = f'b{b_idx}_img{img_idx}_slc{slc_idx}_pred'

                writer.add_image(raw_img_name, img_tensor=raw_img, global_step=epoch, dataformats='HW')
                writer.add_image(gt_img_name, img_tensor=gt_img, global_step=epoch, dataformats='CHW')
                writer.add_image(pred_img_name, img_tensor=pred_img, global_step=epoch, dataformats='CHW')

        with torch.no_grad():
            model.eval()
            print()
            print("validating....")
            for i, (x, y) in enumerate(valid_generator):
                if not args.cpu:
                    x = x.cuda()
                    y = y.cuda()
                y = y.float()

                if args.d == 2:
                    # Input dimensions
                    B, M, _, S, _ = x.shape
                    _, C, _, _, _ = y.shape 
                    H, W = (128,128)
                    # Combine batch and slice dim
                    x = x.permute(0,3,1,2,4).flatten(0,1)  # B x M x H x S x W -> B*S x M x H x W
                    y = y.permute(0,3,1,2,4).flatten(0,1)
                    # Scale
                    x = f.interpolate(x, size=(H,W))
                    y = f.interpolate(y, size=(H,W))

                pred = model(x)

                if args.d == 2:
                    x = x.reshape(B,S,M,H,W).permute(0,2,3,1,4)
                    y = y.reshape(B,S,C,H,W).permute(0,2,3,1,4)
                    pred = f.sigmoid(pred.reshape(B,S,C,H,W).permute(0,2,3,1,4))  # Also apply sigmoid becauce the 2D model doesn't

                loss = criterion(pred, y)
                valid_losses.append(round(loss.item(),4))

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        print("Epoch {}, validation loss is {:.4f}, training loss is {:.4f}".format(epoch + 1, valid_loss,
                                                                                    train_loss))
        if valid_loss < best_loss:  # Saves only best epoch
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),  # only save encoder
                'optimizer_state_dict': optimizer.state_dict()
            }, run_dir + ".pt")
            print("Saving model ", run_dir + ".pt")
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".format(best_loss,
                                                                                                      num_epoch_no_improvement))
            num_epoch_no_improvement += 1
            if num_epoch_no_improvement == args.patience:
                print("Early Stopping")
                break

        if args.tensorboard:
            writer.add_scalar('loss/train', train_loss, epoch)  # Write train loss on tensorboard
            writer.add_scalar('loss/val', valid_loss, epoch)  # Write val loss on tensorboard
            if epoch == 0:  # Only on the first iteration, write model graph on tensorboard
                if args.d == 2:
                    image = image.permute(0,3,1,2,4).flatten(0,1)
                writer.add_graph(model, image)

        sys.stdout.flush()

    return model, writer


def test_segmentation(args, dataloader, in_channels, n_classes, finetuned_model=None, writer=None):

    test_generator = dataloader['test']

    if finetuned_model is not None:  # If model object is directly inserted
        model = finetuned_model
    else:  # If not, load weights from file
        model = prepare_model(args, in_channels, n_classes)
        model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        if not args.cpu:
            model = model.cuda()

    criterion = get_loss(args.n)

    test_loss_arr = []
    test_dice_arr = []
    if args.n == 'brats':
        test_dice_arr = []
        test_dice_wt_arr = []
        test_dice_tc_arr = []
        test_dice_et_arr = []
        
    with torch.no_grad():
        model.eval()
        print("Testing....")
        for i, (x, y) in enumerate(test_generator):
            if not args.cpu:
                x = x.cuda()
                y = y.cuda()
            y = y.float()

            if args.d == 2:
                # Input dimensions
                B, M, _, S, _ = x.shape
                _, C, _, _, _ = y.shape 
                H, W = (128,128)
                # Combine batch and slice dim
                x = x.permute(0,3,1,2,4).flatten(0,1)  # B x M x H x S x W -> B*S x M x H x W
                y = y.permute(0,3,1,2,4).flatten(0,1)
                # Scale
                x = f.interpolate(x, size=(H,W))
                y = f.interpolate(y, size=(H,W))

            pred = model(x)

            if args.d == 2:
                x = x.reshape(B,S,M,H,W).permute(0,2,3,1,4)
                y = y.reshape(B,S,C,H,W).permute(0,2,3,1,4)
                pred = f.sigmoid(pred.reshape(B,S,C,H,W).permute(0,2,3,1,4))  # Also apply sigmoid becauce the 2D model doesn't

            loss = criterion(pred, y)

            if args.n == 'brats':
                test_dice_wt = dice_coeff(pred[:,0], y[:,0])
                test_dice_tc = dice_coeff(pred[:,1], y[:,1])
                test_dice_et = dice_coeff(pred[:,2], y[:,2])
                test_dice = (test_dice_wt + test_dice_tc + test_dice_et) / 3
                test_dice_wt_arr.append(test_dice_wt)
                test_dice_tc_arr.append(test_dice_tc)
                test_dice_et_arr.append(test_dice_et)
                test_dice_arr.append(test_dice)
            elif args.n == 'lits':
                test_dice = dice_coeff(pred, y)
                test_dice_arr.append(test_dice)

            test_loss_arr.append(round(loss.item(),4))

        # logging
        avg_test_loss = np.average(test_loss_arr)
        avg_test_dice = np.average(test_dice_arr)  # average across batches
        if args.n == 'brats':
            avg_test_dice_wt = np.average(test_dice_wt_arr)
            avg_test_dice_tc = np.average(test_dice_tc_arr)
            avg_test_dice_et = np.average(test_dice_et_arr)

        print("Test dice coefficient is {:.4f} . Test loss is {:.4f}".format(avg_test_dice, avg_test_loss))

        if args.tensorboard:
            writer.add_scalar('loss/test', avg_test_loss)  
            writer.add_scalar('dice/test', avg_test_dice)
            if args.n == 'brats':
                writer.add_scalar('dice_wt/test', avg_test_dice_wt)
                writer.add_scalar('dice_tc/test', avg_test_dice_tc)
                writer.add_scalar('dice_et/test', avg_test_dice_et)

        sys.stdout.flush()

    return writer


def train_brats_segmentation(args, dataloader, run_dir, writer=None):
    return train_segmentation(args, dataloader, 4, 3, run_dir, writer)


def test_brats_segmentation(args, dataloader, finetuned_model=None, writer=None):
    return test_segmentation(args, dataloader, 4, 3, finetuned_model, writer)


def train_lits_segmentation(args, dataloader, run_dir, writer=None):
    return train_segmentation(args, dataloader, 1, 1, run_dir, writer)


def test_lits_segmentation(args, dataloader, finetuned_model=None, writer=None):
    return test_segmentation(args, dataloader, 1, 1, finetuned_model, writer)