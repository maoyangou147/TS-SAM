# TS-SAM: Fine-tuning Segment Anything Model For DownStream Tasks

Yang Yu, Chen Xu, Kai Wang*

NanKai University

In Proceedings of the IEEE/CVF International Conference on Multimedia and Expo

Paper link: 
<a href='https://www.arxiv.org/abs/2408.01835'><img src='https://img.shields.io/badge/ArXiv-2408.01835-red' /></a> 

Update on 12 April: This paper is scheduled for oral presentation in ICME 2024

Update on 13 March: This paper is accepted by ICME 2024. 

## Directory Structure

```
TS-SAM/
│   └── .gitignore
│   └── LICENSE
│   └── README.md
│   └── requirements.txt
│   └── sod_metric.py
│   └── test.py
│   └── train.py
│   └── utils.py
│   └── __init__.py
└── configs/
│   │   └── cod-tssam-vit-b.yaml
│   │   └── cod-tssam-vit-h.yaml
│   │   └── sd-tssam-vit-b.yaml
│   │   └── sd-tssam-vit-h.yaml
│   │   └── sod-tssam-vit-b.yaml
│   │   └── sod-tssam-vit-h.yaml
└── datasets/
└── models/
│   │   └── block.py
│   │   └── bn_helper.py
│   │   └── iou_loss.py
│   │   └── models.py
│   │   └── sam.py
│   │   └── __init__.py
│   └── mmseg/
│   │   └── apis/
│   │   └── core/
│   │   └── datasets/
│   │   └── models/
│   │   │   │   └── builder.py
│   │   │   │   └── __init__.py
│   │   │   └── losses/
│   │   │   └── sam/
│   │   │   │   │   └── common.py
│   │   │   │   │   └── feature_fusion_decoder.py (TS-SAM Decoder)
│   │   │   │   │   └── image_encoder.py
│   │   │   │   │   └── image_encoder_ts.py (TS-SAM Encoder)
│   │   │   │   │   └── mask_decoder.py
│   │   │   │   │   └── prompt_encoder.py
│   │   │   │   │   └── sam.py
│   │   │   │   │   └── transformer.py
│   │   │   │   │   └── __init__.py
│   │   │   └── utils/
│   │   └── ops/
│   │   └── utils/
└── pretrained/
│   │   └── .gitignore
```



## Environment Setup
This code was implemented with Python 3.8 and PyTorch 1.13.0. You can install all the requirements via:
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
# install mmcv
pip install -U openmim
mim install mmcv==1.7.0
```


## Dataset Download
### Camouflaged Object Detection
- **[COD10K](https://github.com/DengPingFan/SINet/)**
- **[CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)**
- **[CHAMELEON](https://www.polsl.pl/rau6/datasets/)**
- **[NC4K](https://drive.google.com/file/d/1kzpX_U3gbgO9MuwZIWTuRVpiB7V6yrAQ/view?usp=sharing)**

### Shadow Detection
- **[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN)**

### Salient Object Detection
- **[DUTS](http://saliencydetection.net/duts/#orgf319326)**
- **[ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)**
- **[OMRON](http://saliencydetection.net/dut-omron/#org2daba2e)**
- **[HKU-IS](https://i.cs.hku.hk/~yzyu/research/deep_saliency.html)**
- **[PASCAL-S](https://cbs.ic.gatech.edu/salobj/)**
### Polyp Segmentation - Medical Applications
- **[Kvasir](https://datasets.simula.no/kvasir-seg/)**



## Quick Start
1. Download the dataset.
2. Download the original [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything) checkpoint and put it in `./pretrained`.
3. Set the dataset path and pretrained model path in the yaml file in `./configs`
4. Training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 train.py --config [CONFIG_PATH]
```
## Config
1. cod-tssam-vit-h.yaml: The configuration for Camouflaged Object Detection(COD), using SAM vit-h as the backbone for training.
2. sd-tssam-vit-h.yaml: The configuration for Shadow Detection(SD), using SAM vit-h as the backbone for training.
3. sod-tssam-vit-h.yaml: The configuration for Salient Object Detection(SOD), using SAM vit-h as the backbone for training.

## Train
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 train.py --config [CONFIG_PATH]
```

## Test
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 1 test.py --config [CONFIG_PATH] --model [CHECKPOINT_PATH] --save True
```

## Weights
https://drive.google.com/drive/folders/1dQJiWONDSTrUKCkDLktxaabzgim0OfsQ?usp=drive_link


## Citation

If you find our work useful in your research, please consider citing:

```
@article{yu2024ts,
  title={TS-SAM: Fine-Tuning Segment-Anything Model for Downstream Tasks},
  author={Yu, Yang and Xu, Chen and Wang, Kai},
  journal={arXiv preprint arXiv:2408.01835},
  year={2024}
}
```
## Contact
f you have any questions, please feel free to contact us via email `2110598@mail.nankai.edu.cn` or WeChat `ChenXu2230744290`.

## Acknowledgements
The part of the code is derived from SAM-adapter   <a href='https://github.com/tianrun-chen/SAM-Adapter-PyTorch/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>.

