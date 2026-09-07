# Source-Free Low-Light Image Enhancement via Counterfactual-Inspired and Lighting Debiasing (Under review)

**~~Note:** As the paper is currently under major revision minor revision (TCSVT 26), only the inference code is released. The complete training code will be made publicly available upon acceptance.~~

The training code and some interesting observations are being organized.

# More experimental results.

<img title="" src=".\structured_noise_responses.png" alt="" width="626" data-align="center">

## **Additional Comparison Methods**

**Compared with 2026 methods**. We have added a comparison with Multinex, a supervised method published at CVPR 2026. As shown in Tab. A1, the latest supervised methods still fail to generalize well to unseen target domains.

**Supplementary comparison with unified image restoration methods.** To address whether recent unified (all-in-one) image restoration frameworks can handle the cross-domain low-light enhancement problem studied in our paper, we consider the Real → EndoVis17 and Real → EndoVis18 transfer scenarios, i.e., transferring a model pretrained on LOL-v2-real (daily scenes) to medical endoscopic scenes. We evaluate three representative open-source methods — DiffUIR (CVPR'24), AdaIR (ICLR'25), and DFPIR (CVPR'25) — and compare them with our adapted model (COLD, built upon Retinexformer) as well as two reference baselines: the non-adapted source model (Source only) and the model trained with labeled target-domain data (Target only, serving as the oracle). 

All unified methods are directly deployed with their official pretrained weights , without any fine-tuning or adaptation, i.e., the source-only setting, which is consistent with the deployment of our source model pretrained on LOL-v2-real. The low-light test images of EndoVis17 and EndoVis18 are fed into each model, and PSNR and SSIM are computed against the corresponding normal-light references on the full test sets, following exactly the same evaluation protocol as in the manuscript.

**Table A1. Comparison with recent supervised and unified image restoration methods on Real → EndoVis17 and Real → EndoVis18. All unified methods are directly deployed with their official pretrained weights.**

| Method               | Venue   | Real → EndoVis17 PSNR↑ | Real → EndoVis17 SSIM↑ | Real → EndoVis18 PSNR↑ | Real → EndoVis18 SSIM↑ |
|:--------------------:|:-------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|
| Multinex             | CVPR'26 | 15.29                  | 0.5409                 | 15.31                  | 0.5583                 |
| DiffUIR              | CVPR'24 | 15.45                  | 0.4890                 | 16.17                  | 0.5681                 |
| AdaIR                | ICLR'25 | 13.86                  | 0.4122                 | 12.83                  | 0.4222                 |
| DFPIR                | CVPR'25 | 13.77                  | 0.5145                 | 14.13                  | 0.5788                 |
| **Ours (COLD)**      | -       | **21.25**              | **0.8710**             | **20.18**              | **0.8613**             |
| Target only (Oracle) | -       | 36.70                  | 0.9683                 | 33.72                  | 0.9626                 |

Although the compared unified restoration methods are trained on diverse degradation types and large-scale datasets, they still suffer from severe performance degradation when transferred from daily scenes to unseen medical endoscopic scenes, with PSNR values of only 13.77–16.17 dB. Their performance is comparable to that of the non-adapted source model and falls far behind our adapted model. This is because existing unified restoration methods are trained with full supervision on fixed datasets and degradation types, and thus lack the ability to adapt to unseen domains with large distribution shifts. These results demonstrate that broader training-data coverage cannot replace explicit domain adaptation, which further substantiates the necessity of the proposed source-free debiasing strategy.

## Extended experiments on Synthetic → EndoVis17

**Table A2. Further extended evaluation of perceptual quality, color fidelity, and downstream segmentation performance in the medical endoscopy scenario.**

| Setting               | Method             | LPIPS↓        | $\Delta E$ ↓ | IoU↑ (%)     |
|:---------------------:|:------------------:|:-------------:|:------------:|:------------:|
| Synthetic → EndoVis17 | Source only        | 0.3104        | 19.82        | 24.75        |
|                       | SAME               | 0.2907        | 16.77        | 0.70         |
|                       | **Ours**           | **0.2835**    | **10.32**    | **33.89**    |
|                       | <u>Target only</u> | <u>0.0270</u> | <u>1.52</u>  | <u>73.91</u> |

# EndoVis Inference Code

This package contains the testing code and adapted RetinexFormer checkpoints for
the following source-free domain adaptation settings:

- LOL-v2-real to EndoVis17
- LOL-v2-real to EndoVis18

Only inference-related code and model weights are included in this release.

## Directory structure

```text
release_endovis_inference/
|-- configs/
|   |-- real_to_endovis17.yml
|   `-- real_to_endovis18.yml
|-- weights/
|   |-- real_to_endovis17.pth
|   `-- real_to_endovis18.pth
|-- retinexformer.py
|-- test_from_dataset.py
`-- requirements.txt
```

## Installation

Python 3.8 or later and a CUDA-enabled PyTorch installation are recommended.

```bash
conda create -n endovis-inference python=3.10 -y
conda activate endovis-inference
pip install -r requirements.txt
```

For a system-specific CUDA build, install PyTorch from
<https://pytorch.org/get-started/locally/> first, and then install the remaining
packages with `pip install -r requirements.txt`.

## Data preparation

Paired low-light and reference images must have the same filename. The default
layout is:

```text
datasets/
|-- EndoVis17/
|   |-- low/
|   `-- high/
`-- EndoVis18/
    |-- low/
    `-- high/
```

Supported image extensions are PNG, JPG/JPEG, TIF/TIFF, and BMP. Datasets are available via [Baidu Netdisk](https://pan.baidu.com/s/1keO0QbOkboKY9ZL0WWvJfg) (access code: `bjws`).

## Evaluation

Run all commands from this directory.

Evaluate the real-to-EndoVis17 model:

```bash
python test_from_dataset.py \
  --opt configs/real_to_endovis17.yml \
  --output_dir results/real_to_endovis17 \
  --device cuda:0
```

Evaluate the real-to-EndoVis18 model:

```bash
python test_from_dataset.py \
  --opt configs/real_to_endovis18.yml \
  --output_dir results/real_to_endovis18 \
  --device cuda:0
```

The script saves enhanced PNG images and reports average PSNR and SSIM. To use
data outside the default layout, override both paths:

```bash
python test_from_dataset.py \
  --opt configs/real_to_endovis18.yml \
  --input_dir /path/to/EndoVis18/low \
  --gt_dir /path/to/EndoVis18/high \
  --output_dir results/real_to_endovis18 \
  --device cuda:0
```

For inference without reference images, pass an empty GT value. Metrics are then
skipped:

```bash
python test_from_dataset.py \
  --opt configs/real_to_endovis18.yml \
  --input_dir /path/to/low_light_images \
  --gt_dir "" \
  --output_dir results/unpaired \
  --device cuda:0
```

Optional flags:

- `--self_ensemble`: use x8 test-time augmentation.
- `--GT_mean`: match each prediction's mean intensity to its reference image.
- `--weights /path/to/model.pth`: override the checkpoint in the YAML file.
- `--device cpu`: run without CUDA; this is considerably slower.

## Acknowledgement

The enhancement backbone and the original testing procedure are based on
[Retinexformer](https://github.com/caiyuanhao1998/Retinexformer):

```bibtex
@inproceedings{cai2023retinexformer,
  title={Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
  author={Cai, Yuanhao and Bian, Hao and Lin, Jing and Wang, Haoqian and Timofte, Radu and Zhang, Yulun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}

@inproceedings{zheng2024selective,
  title={Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model},
  author={Zheng, Dian and Wu, Xiao-Ming and Yang, Shuzhou and Zhang, Jian and Hu, Jian-Fang and Zheng, Wei-shi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
@inproceedings{cui2025adair,
title={Ada{IR}: Adaptive All-in-One Image Restoration via Frequency Mining and Modulation},
author={Yuning Cui and Syed Waqas Zamir and Salman Khan and Alois Knoll and Mubarak Shah and Fahad Shahbaz Khan},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}

@inproceedings{tian2025degradation,
  title={Degradation-Aware Feature Perturbation for All-in-One Image Restoration},
  author={Tian, Xiangpeng and Liao, Xiangyu and Liu, Xiao and Li, Meng and Ren, Chao},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={28165--28175},
  year={2025}
}
```
