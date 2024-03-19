import argparse
import os
import time
import warnings

import torch.backends.cudnn
from torch.utils.tensorboard import SummaryWriter
from train_2d import train_pcrlv2
from train_3d import train_pcrlv2_3d, train_cluster_3d
from data import DataGenerator, get_dataloader
from utils import set_seed
from finetune import train_brats_segmentation, test_brats_segmentation, train_lits_segmentation, test_lits_segmentation


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/home/igogou/data/LUNA16',
                        help='path to dataset')
    parser.add_argument('--model', metavar='MODEL', default='pcrlv2', choices=['cluster','pcrlv2','genesis','imagenet','scratch'], help='choose the model')
    parser.add_argument('--phase', default='pretask', choices=['pretask', 'finetune', 'test'], type=str, help='pretask or finetune or test')
    parser.add_argument('--pretrained', default='encoder', choices=['all', 'encoder', 'none'], type=str, help='all or encoder or none')
    parser.add_argument('--finetune', default='all', choices=['all', 'decoder', 'last'], type=str, help='all or decoder or last')
    parser.add_argument('--b', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='epochs to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--output', default='None', type=str, help='output path')
    parser.add_argument('--n', default='luna', choices=['luna', 'lits', 'brats'], type=str, help='dataset to use')
    parser.add_argument('--d', default=3, type=int, help='3d or 2d to run')
    parser.add_argument('--workers', default=4, type=int, help='num of workers')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='gpu indexs')
    parser.add_argument('--ratio', default=0.8, type=float, help='ratio of data used for pretraining')
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight', default=None, type=str)
    parser.add_argument('--weight_decay', default=1e-4)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--patience', default=None, type=int)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--skip_conn', action='store_true', default=False)
    parser.add_argument('--tensorboard', action='store_true', default=False)
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True

    # Set seed
    set_seed(args.seed)
    print(f'Seed is {args.seed}\n')

    data_loader = get_dataloader(args)

    # Force or assert arguments
    if args.model == 'genesis' or args.model == 'imagenet':
        args.skip_conn == False  # Don't add skip connections because they are already there
    if args.phase == 'finetune' and (args.weight is None or args.pretrained == 'none'):
        assert args.weight is None
        if args.model != 'imagenet':  # Because with imagenet pretrained model, we don't load the weights from a file (weight=None) but we do use pretrained weights
            assert args.pretrained == 'none'
    # Define which models can be pretrained or finetuned
    if args.phase in ['finetune','test']:
        assert args.model in ['cluster', 'pcrlv2', 'genesis', 'imagenet', 'scratch']
    elif args.phase == 'pretask':
        assert args.model in ['cluster', 'pcrlv2']

    # Create logger
    curr_time = str(time.time()).replace(".", "")
    
    if args.phase in ['finetune', 'pretask']:  # If finetune or pretask, use the specified output path
        folder_name = None
        
        if args.phase == 'finetune':
            run_name = f'{args.model}_{args.d}d_{"sc_" if args.skip_conn else ""}pretrain_{args.pretrained}_finetune_{args.finetune}_b{args.b}_e{args.epochs}_lr{"{:f}".format(args.lr).split(".")[-1]}_r{int(args.ratio * 100)}_t{curr_time}'
            pretrain_type = None
            # Pretrain types that use weights
            if args.weight:  
                weight_dir = args.weight.lower()
                pretrain_type = args.model
                if 'luna' in weight_dir:
                    pretrain_type += '_luna_pretrain'
                elif 'brats' in weight_dir:
                    pretrain_type += '_brats_pretrain'
                elif 'lits' in weight_dir:
                    pretrain_type += '_lits_pretrain'
                elif 'chest' in weight_dir:
                    pretrain_type += '_chest_pretrain'
            # Pretrain types that don't use weights
            elif not args.weight:
                if args.model == 'imagenet':
                    pretrain_type = f'imagenet_pretrain'
                elif args.model == 'scratch' or args.pretrained == 'none':
                    pretrain_type = f'scratch_{args.d}d'
            folder_name =  args.n + '_finetune' + '_' + pretrain_type
        
        elif args.phase == 'pretask':
            run_name = f'{args.model}_{args.d}d_{"sc_" if args.skip_conn else ""}pretask_b{args.b}_e{args.epochs}_lr{"{:f}".format(args.lr).split(".")[-1]}_t{curr_time}'
            folder_name = args.model + '_' + args.n + '_pretrain'
        
        if not os.path.exists(os.path.join(args.output,folder_name)):
            os.makedirs(os.path.join(args.output,folder_name))
        run_dir = os.path.join(args.output, folder_name, run_name)
    
    elif args.phase == 'test':  # If test, then just use the path from the loaded weights
        run_dir = args.weight.replace('.pt','') # remove .pt

    writer = None   
    if args.tensorboard or args.vis:  # Create tensorboard writer
        assert args.tensorboard  # args.vis can only be used with args.tensorboard
        print(f'Tensorboard logging at: {run_dir}\n')
        writer = SummaryWriter(run_dir)


    # TASKS

    # PRETASK
        
    # 2D PCRLv2 pretask
    if args.model == 'pcrlv2' and args.phase == 'pretask' and args.d == 2:
        train_pcrlv2(args, data_loader, run_dir)
    
    # 3D PCRLv2 pretask
    elif args.model == 'pcrlv2' and args.phase == 'pretask' and args.d == 3:
        train_pcrlv2_3d(args, data_loader, run_dir, writer=writer)

    # 3D Clustering pretask 
    elif args.model == 'cluster' and args.phase == 'pretask' and args.d == 3:
        train_cluster_3d(args, data_loader, run_dir, writer=writer)
        
    # FINETUNING

    # Finetuning+Testings on BraTS
    elif args.phase in 'finetune' and args.n == 'brats':
        model, writer = train_brats_segmentation(args, data_loader, run_dir, writer=writer)
        writer = test_brats_segmentation(args, data_loader, finetuned_model=model, writer=writer)
    
    # Testing on BraTS
    elif args.phase == 'test' and args.n == 'brats':
        test_brats_segmentation(args, data_loader, writer=writer)

    # Finetuning+Testing on LiTS
    elif args.phase in 'finetune' and args.n == 'lits':
        model, writer = train_lits_segmentation(args, data_loader, run_dir, writer=writer)
        writer = test_lits_segmentation(args, data_loader, finetuned_model=model, writer=writer)

    # Testing on LiTS
    elif args.phase == 'test' and args.n == 'lits':
        assert args.d == 3
        test_lits_segmentation(args, data_loader, writer=writer)
    
    if writer:
        writer.close()

