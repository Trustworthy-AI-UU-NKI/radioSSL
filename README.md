# Self-Supervised Learning for Radiology

## Thesis Description
Document: [link](https://dspace.uba.uva.nl/server/api/core/bitstreams/c6347794-efdf-44b4-ba18-fc39c25edc45/content) \
\
"Self-supervised Learning for Radiology", Ioanna Gogou, University of Amsterdam, 2024 \
\
_Deep neural networks have the potential to revolutionize diagnosis in radiology by enabling faster and more precise interpretation of medical images. Despite the success of supervised learning to train these models, its dependency on large annotated datasets presents a major limitation due to the difficulty in acquiring and labeling medical images. This has led to a growing interest in self-supervised learning, which alleviates this issue by generating an artificial supervisory signal from the data itself. This thesis explores the potential of self-supervised learning for segmenting 3D radiology scans. Inspired by previous spatially dense self-supervised methods that showed great success with 2D natural images, we propose patch-level and pixel-level volumetric clustering as a novel pretext task for representation learning on 3D medical data without labels. We evaluated this approach on brain tumor and liver segmentation under limited annotation conditions. Our findings indicate that, while it did not surpass other state-of-the-art self-supervised methods on average, it demonstrated promising results in certain tasks. Moreover, it enhanced segmentation accuracy compared to purely supervised learning and yielded anatomically meaningful representations. However, challenges such as noisy cluster assignments and foreground-background imbalance were observed, suggesting the need for further refinement._

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
    * Brain Tumor Segmentation (BraTS)
    * Liver Segmentation (LiTS)

## Results
### Patch-level Clustering
 ![alt text](images/patch_cluster.png)
### Pixel-level Clustering
 ![alt text](images/pixel_cluster.png)

## Requirements

## Pipeline Execution
### Pretext Task (Pretraining)
   1. Dataset Preprocessing:
       * Example: \
         _(dataset: BraTS, crop size: 128x128x32, crops have max background pixels: 85%)_ \
         ``python preprocess_cluster_images.py --n brats --input_rows 128 --input_cols 128 --input_deps 32 --data /projects/0/prjs0905/data/BraTS18 --save /path/to/data/BraTS18_proc_128 --bg_max 0.85``
   2. K-Means Training (**Only for pixel-level clustering**)
       * Example: \
         _(dataset: preprocessed BraTS, selected gpu device: {0,1}, batch size: 2, clusters: 10, upsampler: FeatUP)_ \
         ``python preprocess_cluster_kmeans_train.py --n brats --data /path/to/data/BraTS18_proc_128 --gpus 0,1 --b 2 --k 10 --upsampler featup``
   3. K-Means Prediction / Ground Truth Generation (**Only for pixel-level clustering**)
       * Example: \
         _(dataset: preprocessed BraTS, selected gpu devices: {0,1}, batch size: 2, clusters: 10, upsampler: FeatUP)_ \
         ``python preprocess_cluster_kmeans_predict.py --n brats --data /projects/0/prjs0905/data/BraTS18_proc_128 --gpus 0,1 --b 2 --k 10 --upsampler featup --centroids /path/to/data/BraTS18_proc_128/kmeans_centroids_k10_featup.npy``
   4. Pretext Task Training
      * Example Pixel-level Clustering: \
        _(dataset: preprocessed BraTS, dimensions: 3D, gpu devices: {0,1}, batch size: 8, epochs: 150, learnining rate: 4e-3, clusters: 10, clustering loss: SwapCE, upsampler: FeatUp, visualize clusters: True)_ \
        ``python main.py --data /path/to/data/BraTS18_proc_128 --model cluster --n brats --d 3 --phase pretask --gpus 0,1 --b 8 --epochs 240 --lr 4e-3 --k 10 --cluster_loss swav --upsampler featup --output /path/to/runs/pretrain --tensorboard --vis``
      * Example Patch-level Clustering: \
         _(dataset: preprocessed BraTS, dimensions: 3D, gpu devices: {0,1}, batch size: 16, epochs: 240, learnining rate: 2e-3, clusters: 50, clustering loss: SwapCE, visualize clusters: True)_ \
        ``python main.py --data /path/to/data/BraTS18_proc_64 --model cluster_patch --n brats --d 3 --phase pretask --gpus 0,1 --b 16 --epochs 240 --lr 2e-3 --k 10 --cluster_loss swav --output /path/to/runs/pretrain --tensorboard --vis``
      * Example PCRLv2: \
        _(dataset: preprocessed BraTS, dimensions: 3D, gpu devices: {0,1}, batch size: 16, epochs: 240, learnining rate: 2e-3)_ \
        ``python main.py --data /path/to/data/BraTS2018_proc_64 --model pcrlv2 --n brats --d 3 --gpus 0,1 --b 16 --epochs 240 --lr 2e-3 --output /path/to/runs/pretrain --tensorboard``
### Downstream Task (Finetuning)
   1. Downstream Task Training
      * Example: \
        _(dataset: BraTS, pretrained model: pixel-level clustering, dimensions: 3D, gpu devices: {0,1}, use skip connections: True, batch size: 4, epochs: 300, learnining rate: 2e-3, pretrained part: encoder, finetuning trainable part: encoder and decoder, data ratio for finetuning: 0.4)_ \
        ``python main.py --data /path/to/data/BraTS18 --model cluster --n brats --d 3 --gpus 0,1 --skip_conn --phase finetune --pretrained encoder --finetune all --ratio 0.4 --lr 1e-3 --b 4 --epochs 300 --output /path/to/runs/finetune --tensorboard --vis --weight /path/to/runs/pretrain/cluster_brats_pretrain/cluster_3d_k10_swav_pretask_b4_e150_lr004000_t171784351201312.pt``

   3. Downstream Task Testing
      * Example: \
        _(dataset: BraTS, pretrained model: pixel-level clustering, dimensions: 3D, gpu devices: {0,1}, use skip connections: True, batch size: 4)_ \
        ``python main.py --data /path/to/data/BraTS18 --model cluster --n brats --d 3 --gpus 0,1 --phase test --skip_conn --b 4 --weight /path/to/runs/finetune/brats_finetune_cluster_brats_pretrain/cluster_3d_k10_sc_pretrain_encoder_finetune_all_b4_e300_lr001000_r40_t17179782752213416.pt --tensorboard
``

## Model Weights
### Pretrained Weights
### Finetuned Weights

## Aknowledgmenets
We thank the authors of the following repositories, parts of which were used for this project: [RL4M/PCRLv2](https://github.com/RL4M/PCRLv2), [MrGiovanni/ModelsGenesis](https://github.com/MrGiovanni/ModelsGenesis)
