# Radiology Self-Supervised Learning

## Thesis Document
[link](https://dspace.uba.uva.nl/server/api/core/bitstreams/c6347794-efdf-44b4-ba18-fc39c25edc45/content)

## Datasets
* BraTS-18
* LiTS-17
* LUNA-16


## Learning Tasks
* Pretext Tasks: \
  _(Performed using datasets BraTS, LiTS, LUNA)_
    * None (downstream task training from scratch)
    * Pixel-level Clustering (ours)
    * Patch-level Clustering (ours)
    * [PCRLv2](https://arxiv.org/abs/2301.00772)
    * [ModelsGenesis](https://arxiv.org/abs/1908.06912)
 * Downstream Tasks:
    *  Brain Tumor Segmentation (BraTS)
    * Liver Segmentation (LiTS)

## Requirements

## Pipeline Execution
### Pretext Task (Pretraining)
   1. Dataset Preprocessing:
       * Example: \
         _(dataset: BraTS, crop size: 128x128x32, crops have max background pixels: 85%)_ \
         ``python preprocess_cluster_images.py --n brats --input_rows 128 --input_cols 128 --input_deps 32 --data /projects/0/prjs0905/data/BraTS18 --save /path/to/data/BraTS18_proc_128 --bg_max 0.85``
   2. K-Means Training (**Only for pixel-level clustering**)
       * Example: \
         _(dataset: preprocessed BraTS, selected gpu device: {0,1}, batch size: 2, clusters: 50, upsampler: FeatUP)_ \
         ``python preprocess_cluster_kmeans_train.py --n brats --data /path/to/data/BraTS18_proc_128 --gpus 0,1 --b 2 --k 50 --upsampler featup``
   3. K-Means Prediction / Ground Truth Generation (**Only for pixel-level clustering**)
       * Example: \
         _(dataset: preprocessed BraTS, selected gpu devices: {0,1}, batch size: 2, clusters: 50, upsampler: FeatUP)_ \
         ``python preprocess_cluster_kmeans_predict.py --n brats --data /projects/0/prjs0905/data/BraTS18_proc_128 --gpus 0,1 --b 2 --k 50 --upsampler featup --centroids /path/to/data/BraTS18_proc_128/kmeans_centroids_k50_featup.npy``
   4. Pretext Task Training
      * Example Pixel-level Clustering: \
        _(dataset: preprocessed BraTS, dimensions: 3D, gpu devices: {0,1}, batch size: 8, epochs: 150, learnining rate: 2e-3, clusters: 50, clustering loss: SwapCE, upsampler: FeatUp, visualize clusters: True)_ \
        ``python main.py --data /path/to/data/BraTS18_proc_128 --model cluster --n brats --d 3 --phase pretask --gpus 0,1 --b 8 --epochs 240 --lr 2e-3 --k 50 --cluster_loss swav --upsampler featup --output /path/to/runs/pretrain --tensorboard --vis``
      * Example Patch-level Clustering: \
         _(dataset: preprocessed BraTS, dimensions: 3D, gpu devices: {0,1}, batch size: 16, epochs: 240, learnining rate: 2e-3, clusters: 50, clustering loss: SwapCE, visualize clusters: True)_ \
        ``python main.py --data /path/to/data/BraTS18_proc_64 --model cluster_patch --n brats --d 3 --phase pretask --gpus 0,1 --b 16 --epochs 240 --lr 2e-3 --k 50 --cluster_loss swav --output /path/to/runs/pretrain --tensorboard --vis``
      * Example PCRLv2: \
        _(dataset: preprocessed BraTS, dimensions: 3D, gpu devices: {0,1}, batch size: 16, epochs: 240, learnining rate: 2e-3)_ \
        ``python main.py --data /path/to/data/BraTS2018_proc_64 --model pcrlv2 --n brats --d 3 --gpus 0,1 --b 16 --epochs 240 --lr 2e-3 --output /path/to/runs/pretrain --tensorboard``
### Downstream Task (Finetuning)
   1. Downstream Task Training
      * Example: \
        _(dataset: BraTS, pretrained model: pixel-level clustering, dimensions: 3D, gpu devices: {0,1}, use skip connections: True, batch size: 4, epochs: 300, learnining rate: 2e-3, pretrained part: encoder, finetuning trainable part: encoder and decoder, data ratio for finetuning: 0.4)_ \
        ``python main.py --data /path/to/data/BraTS18 --model cluster --n brats --d 3 --gpus 0,1 --skip_conn --phase finetune --pretrained encoder --finetune all --ratio 0.4 --lr 1e-3 --b 4 --epochs 300 --output /path/to/runs/finetune --tensorboard --vis --weight /path/to/runs/pretrain/cluster_brats_pretrain/cluster_3d_k50_swav_pretask_b8_e150_lr002000_t17187555844558032.pt``

   3. Downstream Task Testing
      * Example: \
        _(dataset: BraTS, pretrained model: pixel-level clustering, dimensions: 3D, gpu devices: {0,1}, use skip connections: True, batch size: 4)_ \
        ``python main.py --data /path/to/data/BraTS18 --model cluster --n brats --d 3 --gpus 0,1 --phase test --skip_conn --b 4 --weight /path/to/runs/finetune/brats_finetune_cluster_brats_pretrain/cluster_3d_k50_sc_pretrain_encoder_finetune_all_b4_e300_lr001000_r40_t17177179383964033.pt --tensorboard
``


## Aknowledgmenets
We thank the authors of the following repositories, parts of which were used for this project: [RL4M/PCRLv2](https://github.com/RL4M/PCRLv2), [MrGiovanni/ModelsGenesis](https://github.com/MrGiovanni/ModelsGenesis)
