# Source-Free Low-Light Image Enhancement via Counterfactual-Inspired and Lighting Debiasing (Under review)

**Note:** As the paper is currently under major revision (TCSVT 26), only the inference code is released. The complete training code will be made publicly available upon acceptance.

# More experimental results.

<img title="" src=".\structured_noise_responses.png" alt="" width="626">

Fig. A1: Responses of three source-domain enhancement models to different low-light Poisson-noise inputs. The different models produce markedly different colors, brightness levels, and textures, indicating that the enhancement results are significantly influenced by source-domain priors.



**Table A1. Comparison with recent unified image restoration methods on Real → EndoVis17 and Real → EndoVis18. All unified methods are directly deployed with their official pretrained weights.**

| Method               | Venue   | Real → EndoVis17 PSNR↑ | Real → EndoVis17 SSIM↑ | Real → EndoVis18 PSNR↑ | Real → EndoVis18 SSIM↑ |
| -------------------- | ------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| InstructIR           | ICLR'24 | XX.XX                  | 0.XXXX                 | XX.XX                  | 0.XXXX                 |
| AdaIR                | ICLR'25 | 13.86                  | 0.4122                 | 12.82                  | 0.4222                 |
| DFPIR                | CVPR'25 | XX.XX                  | 0.XXXX                 | XX.XX                  | 0.XXXX                 |
| **Ours (COLD)**      | -       | **21.25**              | **0.8710**             | **20.18**              | **0.8613**             |
| Target only (Oracle) | -       | 36.70                  | 0.9683                 | 33.72                  | 0.9626                 |

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
```
