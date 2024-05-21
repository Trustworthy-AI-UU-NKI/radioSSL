import argparse
import os
import warnings

from train_2d import train_pcrlv2_2d, train_cluster_2d
from train_3d import train_pcrlv2_3d, train_cluster_3d
from data import get_dataloader
from tools import set_seed, create_logger
from finetune import train_lidc_segmentation, test_lidc_segmentation, train_brats_segmentation, test_brats_segmentation, train_lits_segmentation, test_lits_segmentation


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self Training benchmark')
    parser.add_argument('--data', metavar='DIR', default='/home/igogou/data/LUNA16',
                        help='Path to dataset')
    parser.add_argument('--data_raw', metavar='DIR', default=None,
                        help='Path to unprocessed dataset (This is needed only for the clustering pretask for visualization)')
    parser.add_argument('--model', metavar='MODEL', default='pcrlv2', choices=['cluster','pcrlv2','genesis','imagenet','scratch'], help='Choose the model')
    parser.add_argument('--phase', default='pretask', choices=['pretask', 'finetune', 'test'], type=str, help='Choose phase: pretask or finetune or test')
    parser.add_argument('--pretrained', default='encoder', choices=['all', 'encoder', 'none'], type=str, help='Choose what is pretrained: all or encoder or none')
    parser.add_argument('--finetune', default='all', choices=['all', 'decoder', 'last'], type=str, help='Choose what to finetune: all or decoder or last')
    parser.add_argument('--b', default=16, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Epochs to train')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--output', default='None', type=str, help='Output path')
    parser.add_argument('--n', default='luna', choices=['luna', 'lidc', 'brats', 'lits'], type=str, help='Dataset to use')
    parser.add_argument('--d', default=3, type=int, help='3D or 2D to run')
    parser.add_argument('--workers', default=4, type=int, help='Num of workers')
    parser.add_argument('--gpus', default='0,1,2,3', type=str, help='GPU indices to use')
    parser.add_argument('--cluster_loss', default='ce', choices=['ce', 'swav'], type=str, help='Choose clustering pretraining loss: cross-entropy or swav')
    parser.add_argument('--ratio', default=1, type=float, help='Ratio of data used for pretraining/finetuning.')
    parser.add_argument('--momentum', default=0.9)
    parser.add_argument('--weight', default=None, type=str, help='Diretory to weights to load')
    parser.add_argument('--weight_decay', default=1e-4)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--patience', default=None, type=int)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--skip_conn', action='store_true', default=False, help='To include skip connections in the U-Net or not. Ideally, use False for pretrain and True for finetune')
    parser.add_argument('--k', default=10, type=int, help='Number of clusters for clustering pretask')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='To log on tensorboard or not')
    parser.add_argument('--vis', action='store_true', default=False, help='To visualize by logging prediction images on tensorboard')
    parser.add_argument('--cpu', action='store_true', default=False, help='To run on CPU or not')
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    print()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

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
    writer, run_dir = create_logger(args)

    # TASKS
        
    # 2D PCRLv2 Pretask
    if args.model == 'pcrlv2' and args.phase == 'pretask' and args.d == 2:
        train_pcrlv2_2d(args, data_loader, run_dir, writer=writer)

    # 2D Cluster Pretask
    if 'cluster' in args.model and args.phase == 'pretask' and args.d == 2:
        train_cluster_2d(args, data_loader, run_dir, writer=writer)
    
    # 3D PCRLv2 Pretask
    elif args.model == 'pcrlv2' and args.phase == 'pretask' and args.d == 3:
        train_pcrlv2_3d(args, data_loader, run_dir, writer=writer)

    # 3D Clustering Pretask 
    elif 'cluster' in args.model and args.phase == 'pretask' and args.d == 3:
        train_cluster_3d(args, data_loader, run_dir, writer=writer)

    # Finetuning + Testing
    elif args.phase in 'finetune':

        if args.n == 'lidc':
            writer = train_lidc_segmentation(args, data_loader, run_dir, writer=writer)
        elif args.n == 'brats':
            writer = train_brats_segmentation(args, data_loader, run_dir, writer=writer)
        elif args.n == 'lits':
            writer = train_lits_segmentation(args, data_loader, run_dir, writer=writer)
        
        args.weight = run_dir + '.pt'  # Change weight argument from pretrained weight to finetuned weight for testing
        args.phase = 'test'  # Change phase from finetune to test
        
        if args.n == 'lidc':
            writer = test_lidc_segmentation(args, data_loader, writer=writer)
        elif args.n == 'brats':
            writer = test_brats_segmentation(args, data_loader, writer=writer)
        elif args.n == 'lits':
            writer = test_lits_segmentation(args, data_loader, writer=writer)

    # Testing
    elif args.phase == 'test':

        if args.n == 'lidc':
            test_lidc_segmentation(args, data_loader, writer=writer)
        elif args.n == 'brats':
            test_brats_segmentation(args, data_loader, writer=writer)
        elif args.n == 'lits':
            test_lits_segmentation(args, data_loader, writer=writer)
    
    if writer:
        writer.close()
