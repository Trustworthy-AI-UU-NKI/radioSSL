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
    loader_name = model + '_' + args.n + '_' + phase
    dataloader = getattr(generator, loader_name)()
    return dataloader


class DataGenerator:

    def __init__(self, args):
        self.args = args

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
        spatial_transforms = [torchio.transforms.RandomFlip(),
                              torchio.transforms.RandomAffine(),
                              ]
        spatial_transforms = torchio.transforms.Compose(spatial_transforms)
        local_transforms = [torchio.transforms.RandomBlur(),
                      torchio.transforms.RandomNoise(),
                      torchio.transforms.RandomGamma(),
                      torchio.transforms.ZNormalization()
                      ]
        local_transforms = torchio.transforms.Compose(local_transforms)
        global_transforms = [torchio.transforms.RandomBlur(),
                             torchio.transforms.RandomNoise(),
                             torchio.transforms.RandomGamma(),
                             torchio.transforms.RandomSwap(patch_size=(8, 4, 4)),
                             torchio.transforms.ZNormalization()
                             ]
        global_transforms = torchio.transforms.Compose(global_transforms)

        train_ds = LunaPretask(args, x_train, train=True, transform=spatial_transforms,
                                     global_transforms=global_transforms, local_transforms=local_transforms)
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
        spatial_transforms = []  # Removed Flip and Affine transform (TODO: Maybe put them back if it doesnt affect mask alignment)

        spatial_transforms = torchio.transforms.Compose(spatial_transforms)
        local_transforms = [
                    #   torchio.transforms.RandomBlur(),
                    #   torchio.transforms.RandomNoise(),
                    #   torchio.transforms.RandomGamma(),
                    #   torchio.transforms.ZNormalization()
                      ]
        local_transforms = torchio.transforms.Compose(local_transforms)
        global_transforms = [
                            #  torchio.transforms.RandomBlur(),
                            #  torchio.transforms.RandomNoise(),
                            #  torchio.transforms.RandomGamma(),
                            #  torchio.transforms.RandomSwap(patch_size=(8, 4, 4)),
                            #  torchio.transforms.ZNormalization()
                             ]
        global_transforms = torchio.transforms.Compose(global_transforms)

        train_ds = LunaPretask(args, x_train, train=True, transform=spatial_transforms,
                                     global_transforms=global_transforms, local_transforms=local_transforms)
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
        spatial_transforms = [torchio.transforms.RandomFlip(),
                              torchio.transforms.RandomAffine(),
                              ]
        spatial_transforms = torchio.transforms.Compose(spatial_transforms)
        local_transforms = [torchio.transforms.RandomBlur(),
                            torchio.transforms.RandomNoise(),
                            torchio.transforms.RandomGamma(),
                            torchio.transforms.ZNormalization()
                            ]
        local_transforms = torchio.transforms.Compose(local_transforms)
        global_transforms = [torchio.transforms.RandomBlur(),
                             torchio.transforms.RandomNoise(),
                             torchio.transforms.RandomGamma(),
                             torchio.transforms.RandomSwap(patch_size=(8, 4, 4)),
                             torchio.transforms.ZNormalization()
                             ]
        global_transforms = torchio.transforms.Compose(global_transforms)

        train_ds = BratsPretask(args, x_train, train=True, transform=spatial_transforms,
                                     global_transforms=global_transforms, local_transforms=local_transforms)
        valid_ds = BratsFinetune(args, x_valid, train=False)

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
        spatial_transforms = []  # Removed Flip and Affine transform (TODO: Maybe put them back if it doesnt affect mask alignment)
        spatial_transforms = torchio.transforms.Compose(spatial_transforms)
        local_transforms = [
                        #    torchio.transforms.RandomNoise(std=0.01) # Removed destructive transforms and Z-norm (TODO: Maybe put them back once we know it works)
                        ]  
        local_transforms = torchio.transforms.Compose(local_transforms)
        global_transforms = [
                            # torchio.transforms.RandomNoise(std=0.01), # TODO: Maybe put them back once we know it works
                        ]  
        global_transforms = torchio.transforms.Compose(global_transforms)

        train_ds = BratsPretask(args, x_train, train=True, transform=spatial_transforms,
                                     global_transforms=global_transforms, local_transforms=local_transforms)
        valid_ds = BratsPretask(args, x_valid, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)

        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        return dataloader

    def pcrlv2_brats_finetune(self):
        args = self.args
        dataloader = {}
        train_list, val_list, test_list = get_brats_list(self.args.data, self.args.ratio)
        train_ds = BratsFineTune(train_list, train=True)
        val_ds = BratsFineTune(val_list, train=False)
        test_ds = BratsFineTune(test_list, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)
        dataloader['train'] = DataLoader(train_ds, batch_size=self.args.b,
                                         num_workers=self.args.workers,
                                         worker_init_fn=seed_worker,
                                         generator=generator,
                                         pin_memory=True,
                                         shuffle=True)
        dataloader['eval'] = DataLoader(val_ds, batch_size=self.args.b,
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

    def pcrlv2_luna_finetune(self):
        luna_valid_txt = 'train_val_txt/luna_finetune_test.txt'
        luna_test_txt = 'train_val_txt/luna_finetune_valid.txt'
        # luna_train_txt = 'train_val_txt/luna_finetune_train.txt'
        args = self.args
        dataloader = {}
        train_fold = [0, 1, 2, 3, 4, 5, 6]
        valid_fold = [7, 8]
        test_fold = [9]
        file_list = get_luna_finetune_list(args.ratio, args.data, train_fold)
        print(len(file_list))
        x_train, x_valid, x_test = get_luna_finetune_nodule(args, train_fold, luna_valid_txt, luna_test_txt,
                                                            suffix='.npy',
                                                            file_list=file_list)
        print(len(x_train))
        train_ds_true, train_ds_false = divide_luna_true_positive(x_train)
        valid_ds_true, valid_ds_false = divide_luna_true_positive(x_valid)
        test_ds_true, test_ds_false = divide_luna_true_positive(x_test)

        train_ds = LunaFineTune(args, train_ds_true, train_ds_false, train=True)
        valid_ds = LunaFineTune(args, valid_ds_true, valid_ds_false, train=False)
        test_ds = LunaFineTune(args, test_ds_true, test_ds_false, train=False, test=True)
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        dataloader['train'] = DataLoader(train_ds, batch_size=args.b,
                                         pin_memory=True, shuffle=True, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['eval'] = DataLoader(valid_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        dataloader['test'] = DataLoader(test_ds, batch_size=args.b,
                                        pin_memory=True, shuffle=False, num_workers=args.workers, worker_init_fn=seed_worker, generator=generator)
        print(f"Training Nodule{len(train_ds)} Valid Nodule {len(valid_ds)}, Test Nodule {len(test_ds)}")
        return dataloader

    def pcrlv2_lits_finetune(self):
        args = self.args
        dataloader = {}

        train_path = os.path.join(args.data, 'train')
        valid_path = os.path.join(args.data, 'val')

        train_ds = LitsFineTune(os.path.join(train_path, 'ct'), os.path.join(train_path, 'seg'), training=True, ratio=self.args.ratio)
        valid_ds = LitsFineTune(os.path.join(valid_path, 'ct'), os.path.join(valid_path, 'seg'), training=False)  
        test_ds = LitsFineTune(os.path.join(valid_path, 'ct'), os.path.join(valid_path, 'seg'), training=False) # Use val==test

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
