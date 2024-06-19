import io
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import random
from copy import deepcopy
import PIL
import cv2

from tools import prepare_model, get_loss, dice_coeff
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as t

from matplotlib import pyplot as plt


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

    grid_pred = []  # Grid for visualizing predictions at each epoch

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

            # Input dimensions
            B, M, H, W, D = image.shape
            _, C, _, _, _ = gt.shape 

            # if args.tensorboard and epoch == 0:  # Only on the first iteration, write model graph on tensorboard
            #     if args.d == 2:
            #         writer.add_graph(model, image.permute(0,3,1,2,4).flatten(0,1))
            #     elif args.d == 3:
            #         writer.add_graph(model, image)

            if args.d == 2: # If model is 2D unet, then combine batch and slice dimension and scale input to power of 2
                # Input dimensions
                B, M, H, W, D = image.shape
                _, C, _, _, _ = gt.shape 
                # Combine batch and slice dim
                image = image.permute(0,4,1,2,3).reshape(B*D,M,H,W)  # B x M x H x W x D -> B*D x M x H x W
                gt = gt.permute(0,4,1,2,3).reshape(B*D,C,H,W)

            pred = model(image) 

            if args.d == 2: # If 2D unet, then revert to original dims
                image = image.reshape(B,D,M,H,W).permute(0,2,3,4,1)
                gt = gt.reshape(B,D,C,H,W).permute(0,2,3,4,1)
                pred = f.sigmoid(pred.reshape(B,D,C,H,W).permute(0,2,3,4,1))  # Also apply sigmoid becauce the 2D model doesn't

            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(round(loss.item(), 2))
            if (iteration + 1) % 5 == 0:
                print('Epoch [{}/{}], iteration {}, Loss:{:.6f}, {:.6f}'
                      .format(epoch + 1, args.epochs, iteration + 1, loss.item(), np.average(train_losses)))
                sys.stdout.flush()

            # Plot predictions on tensorboard
            b_idx = 0
            if args.vis and iteration==b_idx and epoch % 5 == 0:
                mod_idx = 0
                c_idx = 0
                img_idx = 0
                slc_idx = image.shape[4] // 2

                image = image[img_idx,mod_idx,:,:,slc_idx].cpu().detach().numpy()
                gt = gt[img_idx,:,:,:,slc_idx].cpu().detach().numpy()
                pred = pred[img_idx,:,:,:,slc_idx].cpu().detach().numpy()

                image_name = f'b{b_idx}_img{img_idx}_slc{slc_idx}_raw'
                gt_name = f'b{b_idx}_img{img_idx}_slc{slc_idx}_gt'
                pred_name = f'b{b_idx}_img{img_idx}_slc{slc_idx}_pred'

                writer.add_image(image_name, img_tensor=image, global_step=epoch, dataformats='HW')
                writer.add_image(gt_name, img_tensor=gt, global_step=epoch, dataformats='CHW')
                writer.add_image(pred_name, img_tensor=pred, global_step=epoch, dataformats='CHW')

        with torch.no_grad():
            model.eval()
            print()
            print("Validating....")

            # Hyperparameters for grid visualization
            N = 8  # Grid row/col size
            n_epochs = min(N,args.epochs) # The number of epochs to sample for the grid (N or all epochs if total less than N)
            step_epochs =  args.epochs // n_epochs # Every how many epochs to sample

            for i, (x, y) in enumerate(valid_generator):

                if not args.cpu:
                    x = x.cuda()
                    y = y.cuda()
                y = y.float()

                # Input dimensions
                B, M, H, W, D = x.shape
                _, C, _, _, _ = y.shape 

                if args.d == 2:
                    # Combine batch and slice dim
                    x = x.permute(0,4,1,2,3).flatten(0,1)  # B x M x H x W x D -> B*D x M x H x W
                    y = y.permute(0,4,1,2,3).flatten(0,1)

                pred = model(x)

                if args.d == 2:
                    x = x.reshape(B,D,M,H,W).permute(0,2,3,4,1)
                    y = y.reshape(B,D,C,H,W).permute(0,2,3,4,1)
                    pred = f.sigmoid(pred.reshape(B,D,C,H,W).permute(0,2,3,4,1))  # Also apply sigmoid because the 2D model doesn't

                # Calculate loss
                loss = criterion(pred, y)
                valid_losses.append(round(loss.item(),4))

                # Gather predictions to visualize on grid
                if args.vis and (epoch % step_epochs == 0) and (epoch / step_epochs) <= n_epochs and i==0:  # Only visualize batch 0 (i) for the sampled epochs
                    n_images = min(N,args.b)  # The number of images to sample from for the grid (N or all images if total less than N)
                    if epoch == 0:  # If epoch 0, add the input images and ground truth as two first rows of the grid
                        for img_idx in range(n_images):
                            if args.n == 'brats':
                                slice_idx = [100, 40, 55, 85]  # TODO: Works only for batch size = 4
                            else:
                                slice_idx = [60, 15, 32, 52]  # TODO: Works only for batch size = 4
                            # Input
                            x_i = x[img_idx,0,:,:,slice_idx[img_idx]]                   
                            x_i = (x_i - x_i.min())/(x_i.max() - x_i.min())  # Min-max norm input images
                            x_i = x_i.repeat((3,1,1)).permute(1,2,0)  # Convert to RGB and move channel dim to the end
                            x_i = x_i.cpu().detach().numpy()

                            # Ground truth segmentation mask
                            y_i = y[img_idx,:,:,:,slice_idx[img_idx]] 
                            y_i = y_i.permute(1,2,0)
                            y_i = y_i.cpu().detach().numpy()
                            if y_i.shape[-1] != 3:  # If not already RGB, convert to RGB with red color for mask
                                y_i = np.concatenate([y_i,np.zeros(y_i.shape),np.zeros(y_i.shape)], axis=2)
                            y_i = y_i.astype(np.float32)
                            
                            # Apply segmentation mask on image
                            if args.n == 'brats':
                                y_i[np.all(y_i==[1,0,0], axis=-1)] = [0,1,0]  # Convert red to green (WT)
                                y_i[np.all(y_i==[1,1,1], axis=-1)] = [0,0,1]  # Convert white to blue (ET)
                                y_i[np.all(y_i==[1,1,0], axis=-1)] = [1,0,0]  # Convert yellow to red (TC)
                            alpha_x_i = 1 - x_i
                            mask_i = np.expand_dims((np.sum(y_i,axis=2)!=0),2).repeat(3, axis=2)
                            blend_x_y_i = cv2.addWeighted(x_i, 0.4, y_i, 0.6, 0)  # Add trasnparency to seg. mask
                            masked_x_y_i = np.where(mask_i!=[0,0,0], blend_x_y_i, x_i)  # Apply seg. mask
                            x_y_i = x_i * alpha_x_i + masked_x_y_i * (1 - alpha_x_i)  # Reapply shadows
                            grid_pred.append(x_y_i)
                    else:
                        for img_idx in range(n_images): # If next epochs, add the predictions for each image at the current epoch as the next row
                            pred_i = pred[img_idx,:,:,:,slice_idx[img_idx]]
                            pred_i = pred_i.permute(1,2,0)
                            pred_i = pred_i.cpu().detach().numpy()
                            if args.n == 'brats':  # We have 3 classes for BraTS
                                temp_pred_i = np.zeros(pred_i.shape)
                                temp_pred_i[:,:,1] = np.fmax(0, pred_i[:,:,0] - pred_i[:,:,1]) # WT  (Because the more red {TC} we have the less green {WT} we want)
                                temp_pred_i[:,:,2] = np.fmax(0, pred_i[:,:,2])  # ET
                                temp_pred_i[:,:,0] = np.fmax(0,pred_i[:,:,1] - pred_i[:,:,2]) # TC  (Because the more blue {ET} we have the less red {TC}} we want)
                                pred_i = temp_pred_i
                            else:  # We have 1 class for LiTS, so convert mask to RGB
                                pred_i = np.repeat(pred_i,3,axis=2)

                            grid_pred.append(pred_i)
                
            # Plot grid of predictions for sampled epochs up to now
            if args.vis and (epoch % step_epochs == 0) and (epoch / step_epochs) <= n_epochs:
                n_cols = min(N,args.b)
                n_rows = len(grid_pred) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15*(n_rows/n_cols)))
                for i, ax in enumerate(axes.flat):
                    ax.imshow(grid_pred[i]) 
                    ax.axis('off')  # Turn off axis labels
                    if i % n_cols:
                        ax.set_ylabel(f'Epoch {epoch}', rotation=0, size='large')
                plt.tight_layout()  # Adjust spacing between subplots
                # Save grid to buffer and then log on tensorboard
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                buf.seek(0)
                grid = PIL.Image.open(buf)
                grid = t.pil_to_tensor(grid)
                writer.add_image(f'img/val/grid', img_tensor=grid, global_step=epoch)

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
                'state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, run_dir + ".pt")
            print("Saving model ", run_dir + ".pt\n")
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}\n".format(best_loss,
                                                                                                      num_epoch_no_improvement))
            num_epoch_no_improvement += 1
            if num_epoch_no_improvement == args.patience:
                print("Early Stopping")
                break

        if args.tensorboard:
            writer.add_scalar('loss/train', train_loss, epoch)  # Write train loss on tensorboard
            writer.add_scalar('loss/val', valid_loss, epoch)  # Write val loss on tensorboard

        sys.stdout.flush()

    return writer


def test_segmentation(args, dataloader, in_channels, n_classes, writer=None):

    test_generator = dataloader['test']

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
                B, M, H, W, D = x.shape
                _, C, _, _, _ = y.shape 

                # Combine batch and slice dim
                x = x.permute(0,4,1,2,3).flatten(0,1)  # B x M x H x W x D -> B*D x M x H x W
                y = y.permute(0,4,1,2,3).flatten(0,1)

            pred = model(x)

            if args.d == 2:
                x = x.reshape(B,D,M,H,W).permute(0,2,3,4,1)
                y = y.reshape(B,D,C,H,W).permute(0,2,3,4,1)
                pred = f.sigmoid(pred.reshape(B,D,C,H,W).permute(0,2,3,4,1))  # Also apply sigmoid becauce the 2D model doesn't

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
            else:
                test_dice = dice_coeff(pred, y)
                test_dice_arr.append(test_dice)

            test_loss_arr.append(round(loss.item(),4))

        # logging
        avg_test_loss = np.average(test_loss_arr)
        avg_test_dice = np.average(test_dice_arr) 
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


def train_lidc_segmentation(args, dataloader, run_dir, writer=None):
    return train_segmentation(args, dataloader, 1, 1, run_dir, writer)


def test_lidc_segmentation(args, dataloader, finetuned_model=None, writer=None):
    return test_segmentation(args, dataloader, 1, 1, writer)


def train_brats_segmentation(args, dataloader, run_dir, writer=None):
    return train_segmentation(args, dataloader, 4, 3, run_dir, writer)


def test_brats_segmentation(args, dataloader, finetuned_model=None, writer=None):
    return test_segmentation(args, dataloader, 4, 3, writer)


def train_lits_segmentation(args, dataloader, run_dir, writer=None):
    return train_segmentation(args, dataloader, 1, 1, run_dir, writer)


def test_lits_segmentation(args, dataloader, finetuned_model=None, writer=None):
    return test_segmentation(args, dataloader, 1, 1, writer)
