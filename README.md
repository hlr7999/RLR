# RLR

> [ECCV 2024] Learning Unified Reference Representation for Unsupervised Multi-class Anomaly Detection

## Setup

### Environment

We utilize the `Python 3.9` interpreter in our experiments. Install the required packages using the following command:
```bash
pip3 install -r requirements.txt
```

### Datasets

Download [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) or [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) datasets, and organize them in the following file structure (the default structure of MVTec-AD):
```
├── class1
│   ├── ground_truth
│   │   ├── defect1
│   │   └── defect2
│   ├── test
│   │   ├── defect1
│   │   ├── defect2
│   │   └── good
│   └── train
│       └── good
├── class2
...
```

### Pretrained Model

We load the pretrained model weights from local files, as indicated by code `line 57` in file `trainer.py`. 

```python
if 'efficientnet' in self.args.backbone_arch:
    config = efn_cfg(url='', file=f'{self.args.root_path}/pretrained/tf_efficientnet_b6_aa-80ba17e4.pth')
elif 'resnet50' in self.args.backbone_arch:
    config = res_cfg(url='', file=f'{self.args.root_path}/pretrained/wide_resnet50_racm-8234f177.pth')
encoder = timm.create_model(
    self.args.backbone_arch,
    features_only=True,
    pretrained_cfg=config, 
    out_indices=self.args.out_indices,
    pretrained=True
)
```

However, it can be modified to download and load the pretrained model from the network, by simply deleting this config and `pretrained_cfg=config` in `timm.create_model` function.

## Train

Train our RLR with the following command:

```bash
python3 main.py \
        --root_path $your_proj_path \
        --dataset mvtec \ # mvtec or visa
        --data_path $your_data_path \
        --backbone_arch tf_efficientnet_b6 \ # efficientnet or wrideresenet
        --feature_levels 2 \ # 2 or 3
        --out_indices 2 3 \ # 2 3 or 1 2 3
        --feature_jitter 4 \
        --layers 4 \
        --blocks mca nsa \
        --blocks_gate none \
        --batch_size 4 \
        --num_epochs 200 \
        --save_prefix $tag
```

## Test

Test the model with the following command:

```bash
python3 main.py \
        --root_path $your_proj_path \
        --dataset mvtec \ # mvtec or visa
        --data_path $your_data_path \
        --backbone_arch tf_efficientnet_b6 \ # efficientnet or wrideresenet
        --feature_levels 2 \ # 2 or 3
        --out_indices 2 3 \ # 2 3 or 1 2 3
        --layers 4 \
        --blocks mca nsa \
        --blocks_gate none \
        --batch_size 16 \
        --save_prefix $tag \
        --mode test \
        --vis
```

## Decoder

We train additional decoder models to visualize the features, allowing for a more intuitive display of the reconstructed feature effects. It is important to **note** that the decoder is solely used for **feature visualization** and does not participate in the anomaly detection process. Therefore, the decoder is **not mandatory**.

Train the decoder with the following command:

```bash
python3 main.py \
        --root_path $your_proj_path \
        --dataset mvtec \ # mvtec or visa
        --data_path $your_data_path \
        --backbone_arch tf_efficientnet_b6 \ # efficientnet or wrideresenet
        --feature_levels 2 \ # 2 or 3
        --out_indices 2 3 \ # 2 3 or 1 2 3
        --with_decoder \
        --batch_size 16 \
        --num_epochs 100 \
```

Add `--with_decoder` to the aforementioned `Test command` to incorporate the visualization of features in the test results.
