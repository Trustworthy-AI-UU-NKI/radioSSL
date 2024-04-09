from torch.utils.data import DataLoader

from datasets import *
from utils import *
from torchvision import transforms, datasets
import torch
import torchio.transforms
import copy
import numpy


def get_dataloader(args):
    generator = DataGenerator(args)
    # Phase corrections
    if args.phase == 'test':
        phase = 'finetune'  # Because finetune and test use the same dataloader
    else:
        phase = args.phase
    # Model corrections when finetuning
    if args.model in ['genesis', 'imagenet', 'scratch', 'cluster'] and phase == 'finetune':  # Attention: phase and not args.phase
        model = 'pcrlv2'  # Because those models and pcrlv2 use the same dataloader during finetuning (cluster has its own during pretraining)
    else:
        model = args.model
    if args.phase == 'pretask':
        loader_name = model + '_' + args.n + '_' + phase
    else:
        loader_name = args.n + '_' + phase
    dataloader = getattr(generator, loader_name)()
    return dataloader


class DataGenerator:

    def __init__(self, args):
        self.args = args
        
        self.pcrlv2_spatial_transforms = torchio.transforms.Compose([torchio.transforms.RandomFlip(),
                              torchio.transforms.RandomAffine(),
                              ])
        self.pcrlv2_local_transforms = torchio.transforms.Compose([torchio.transforms.RandomBlur(),
                            torchio.transforms.RandomNoise(),
                            torchio.transforms.RandomGamma(),
                            torchio.transforms.ZNormalization()
                            ])
        self.pcrlv2_global_transforms = torchio.transforms.Compose([torchio.transforms.RandomBlur(),
                             torchio.transforms.RandomNoise(),
                             torchio.transforms.RandomGamma(),
                             torchio.transforms.RandomSwap(patch_size=(8, 4, 4)),
                             torchio.transforms.ZNormalization()
                             ])

        self.cluster_spatial_transforms = torchio.transforms.Compose([])
        self.cluster_local_transforms = torchio.transforms.Compose([])
        self.cluster_global_transforms = torchio.transforms.Compose([])


    # PRETASK

    def pcrlv2_luna_pretask(self):
        print('using the reverse_aug pretrain on luna')
        args = self.args
        dataloader = {}
        train_fold = [0, 1, 2, 3, 4, 5, 6]
        valid_fold = [7, 8, 9]
        file_list = get_luna_pretrain_list(args.ratio)
        x_train, x_valid, _ = get_luna_list(args, train_fold, valid_fold, valid_fold, suffix='_global_',
                                            file_list=file_list)
        print(f'Train Images {len(x_train)}, Valid Images {len(x_valid)}')

        train_ds = LunaPretask(args, x_train, train=True, transform=self.pcrlv2_spatial_transforms,
                                     global_transforms=self.pcrlv2_global_transforms, local_transforms=self.pcrlv2_local_transforms)
        valid_ds = LunaPretask(args, x_valid, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        return dataloader

    def cluster_luna_pretask(self):
        print('using the reverse_aug pretrain on luna')
        args = self.args
        dataloader = {}
        train_fold = [0, 1, 2, 3, 4, 5, 6]
        valid_fold = [7, 8, 9]
        pretrain_list = get_luna_pretrain_list(args.ratio)
        x_train, x_valid, _ = get_luna_list(args, train_fold, valid_fold, valid_fold, suffix='_global_',
                                            file_list=pretrain_list)
        # finetune_list = get_luna_finetune_list(0.05)  # Ratio doesn't really matter because we simply use the val dataset for visualization)
        print(f'Train Images {len(x_train)}, Valid Images {len(x_valid)}')

        train_ds = LunaPretask(args, x_train, train=True, transform=self.cluster_spatial_transforms,
                                     global_transforms=self.cluster_global_transforms, local_transforms=self.cluster_local_transforms)
        valid_ds = LunaPretask(args, x_valid, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        return dataloader

    def pcrlv2_brats_pretask(self):
        print('using the reverse_aug pretrain on brats')
        args = self.args
        dataloader = {}

        x_train, x_valid, _ = get_brats_pretrain_list(self.args.data, self.args.ratio, suffix='_global_')
        print(f'Train Images {len(x_train)}, Valid Images {len(x_valid)}')

        train_ds = BratsPretask(args, x_train, train=True, transform=self.pcrlv2_spatial_transforms,
                                     global_transforms=self.pcrlv2_global_transforms, local_transforms=self.pcrlv2_local_transforms)
        valid_ds = BratsPretask(args, x_valid, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        return dataloader

    def cluster_brats_pretask(self):
        print('using the reverse_aug pretrain on brats')
        args = self.args
        dataloader = {}
        
        x_train, x_valid, _ = get_brats_pretrain_list(self.args.data, self.args.ratio, suffix='_global_')
        print(f'Train Images {len(x_train)}, Valid Images {len(x_valid)}')

        train_ds = BratsPretask(args, x_train, train=True, transform=self.cluster_spatial_transforms,
                                     global_transforms=self.cluster_global_transforms, local_transforms=self.cluster_local_transforms)
        valid_ds = BratsPretask(args, x_valid, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        return dataloader

    def pcrlv2_lits_pretask(self):
        args = self.args
        dataloader = {}

        train_path = os.path.join(args.data, 'train', 'ct')
        valid_path = os.path.join(args.data, 'val', 'ct')

        x_train = [os.path.join(train_path,x) for x in os.listdir(train_path) if 'global' in x]
        x_valid = [os.path.join(valid_path,x) for x in os.listdir(valid_path) if 'global' in x]
        print(f'Train Images {len(x_train)}, Valid Images {len(x_valid)}')

        train_ds = LitsPretask(args, x_train, train=True, transform=self.pcrlv2_spatial_transforms,
                                     global_transforms=self.pcrlv2_global_transforms, local_transforms=self.pcrlv2_local_transforms)
        valid_ds = LitsPretask(args, x_valid, train=False) 

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        return dataloader

    def cluster_lits_pretask(self):
        args = self.args
        dataloader = {}

        train_path = os.path.join(args.data, 'train', 'ct')
        valid_path = os.path.join(args.data, 'val', 'ct')

        x_train = [os.path.join(train_path,x) for x in os.listdir(train_path) if 'global' in x]
        x_valid = [os.path.join(valid_path,x) for x in os.listdir(valid_path) if 'global' in x]
        print(f'Train Images {len(x_train)}, Valid Images {len(x_valid)}')

        train_ds = LitsPretask(args, x_train, train=True, transform=self.cluster_spatial_transforms,
                                     global_transforms=self.cluster_global_transforms, local_transforms=self.cluster_local_transforms)
        valid_ds = LitsPretask(args, x_valid, train=False)  

        print(f'Train Images {len(x_train)}, Valid Images {len(x_valid)}')

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        return dataloader
 

    # FINETUNING

    def brats_finetune(self):
        args = self.args
        dataloader = {}
        train_list, val_list, test_list = get_brats_list(self.args.data, self.args.ratio)
        train_ds = BratsFineTune(train_list, train=True)
        valid_ds = BratsFineTune(val_list, train=False)
        test_ds = BratsFineTune(test_list, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=self.args.b,
                                         num_workers=self.args.workers,
                                         worker_init_fn=seed_worker,
                                         generator=generator,
                                         pin_memory=True,
                                         shuffle=True)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=self.args.b,
                                        num_workers=self.args.workers,
                                        worker_init_fn=seed_worker,
                                        generator=generator,
                                        pin_memory=True,
                                        shuffle=False)
        dataloader['test'] = DataLoader(test_ds, batch_size=1, num_workers=self.args.b,
                                        worker_init_fn=seed_worker,
                                        generator=generator,
                                        pin_memory=True,
                                        shuffle=False)
        return dataloader

    def lidc_finetune(self):
        args = self.args
        dataloader = {}
        
        train_list, valid_list, test_list = get_lidc_list(args.ratio, args.data)

        train_ds = LidcFineTune(args, train_list, train=True)
        valid_ds = LidcFineTune(args, valid_list, train=False)
        test_ds = LidcFineTune(args, test_list, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=self.args.b,
                                         num_workers=self.args.workers,
                                         worker_init_fn=seed_worker,
                                         generator=generator,
                                         pin_memory=True,
                                         shuffle=True)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=self.args.b,
                                        num_workers=self.args.workers,
                                        worker_init_fn=seed_worker,
                                        generator=generator,
                                        pin_memory=True,
                                        shuffle=False)
        dataloader['test'] = DataLoader(test_ds, batch_size=1, num_workers=self.args.b,
                                        worker_init_fn=seed_worker,
                                        generator=generator,
                                        pin_memory=True,
                                        shuffle=False)
        print(f"Training Nodule{len(train_ds)} Valid Nodule {len(valid_ds)}, Test Nodule {len(test_ds)}")
        return dataloader

    def lits_finetune(self):
        args = self.args
        dataloader = {}

        train_path = os.path.join(args.data, 'train')
        valid_path = os.path.join(args.data, 'val')

        train_ds = LitsFineTune(os.path.join(train_path, 'ct'), os.path.join(train_path, 'seg'), train=True, ratio=self.args.ratio)
        valid_ds = LitsFineTune(os.path.join(valid_path, 'ct'), os.path.join(valid_path, 'seg'), train=False)  
        test_ds = LitsFineTune(os.path.join(valid_path, 'ct'), os.path.join(valid_path, 'seg'), train=False) # Use val==test

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=self.args.b,
                                         num_workers=self.args.workers,
                                         worker_init_fn=seed_worker,
                                         generator=generator,
                                         pin_memory=True,
                                         shuffle=True)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=self.args.b,
                                        num_workers=self.args.workers,
                                        worker_init_fn=seed_worker,
                                        generator=generator,
                                        pin_memory=True,
                                        shuffle=False)
        dataloader['test'] = DataLoader(test_ds, batch_size=1, num_workers=self.args.b,
                                        worker_init_fn=seed_worker,
                                        generator=generator,
                                        pin_memory=True,
                                        shuffle=False)
        return dataloader
