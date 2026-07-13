# CVSF

Official inference package for **Bridging the Semantic Gap in Extreme Degradation: A Cognition-Vision Synergistic Fusion Framework for Coupled Low-Light Super-Resolution**. This release contains code, configuration files, checkpoint placement instructions, training templates, inference scripts, evaluation scripts, and logs for reproducing the reported restoration settings.

## Overview

The package supports:

- CVSF inference on RELLISUR
- whole-folder metric evaluation
- Stage-1 and Stage-2 training launch templates
- representative logs for compared methods

## Environment

```bash
conda env create -f environment.yml
conda activate CVSF
```

`requirements.txt` lists the main pip packages.

## Dataset

Prepare the RELLISUR dataset and update the meta path in `inference.sh`. The provided example uses the X2 split.

Official RELLISUR links:

```text
Project page: https://vap.aau.dk/rellisur/
Dataset DOI:  https://doi.org/10.5281/zenodo.5234969
Zenodo:       https://zenodo.org/records/5234969
```

```bash
--input_meta="/path/to/RELLISUR_testX2-new.txt"
```

For a minimal runnable example, five paired RELLISUR samples are included under `samples/`, and `inference.sh` uses:

```bash
--input_meta="examples/RELLISUR_testX2_example.txt"
```

Each line in the meta file should contain one ground-truth image path and one low-quality image path:

```text
/path/to/GT.png /path/to/LQ.png
```

Example:

```text
samples/RELLISUR-Dataset/Test/NLHR/X2/00736.png samples/RELLISUR-Dataset/Test/LLLR/00736-5.0.png
```

Expected RELLISUR structure:

```text
RELLISUR-Dataset/
|-- Test/
|   |-- NLHR/X2/
|   |-- NLHR/X4/
|   `-- LLLR/
`-- Train/
```

For X4 experiments, replace the meta file, config/output names, and dataset paths with the corresponding X4 files following the same format.

Stage-1 EEG/image alignment requires the EEG and THINGS-derived image data used by `configs/stage1_alignment.yaml`.

Official THINGS / THINGS-EEG2 links:

```text
THINGS project:       https://things-initiative.org/
THINGS-EEG2 OSF:      https://osf.io/3jk45/
Raw EEG OSF:          https://osf.io/crxs4/
THINGS image set OSF: https://osf.io/y63gw/
HuggingFace mirror:   https://huggingface.co/datasets/gasparyanartur/things-eeg2
```

Example paths after downloading and preprocessing are:

```text
EEG raw:          /path/to/THINGS-EEG2/raw_eeg_data/
EEG preprocessed: /path/to/THINGS-EEG2/preprocessed_eeg_data/
training images:  /path/to/THINGS-EEG2/image_filtering_dataset/training_images
test images:      /path/to/THINGS-EEG2/image_filtering_dataset/test_images
features:         /path/to/THINGS-EEG2/features/
```

Edit `configs/stage1_alignment.yaml` if these paths differ.

Two Stage-1-related files are provided for reproducibility:

```text
checkpoints/stage1/image_40.pth
checkpoints/stage1/eeg_40.pth
embeddings/rellisur_test_aligned_cognition_embeddings.pt
```

`checkpoints/stage1/image_40.pth` is the Stage-1 image encoder / alignment projector checkpoint obtained from Stage-1 contrastive learning. During the default inference pipeline, the degraded input image is passed through this checkpoint to compute the aligned cognition embedding online.

`checkpoints/stage1/eeg_40.pth` is the Stage-1 EEG encoder checkpoint obtained from the same Stage-1 contrastive alignment training. It is provided to document and reproduce the cognition-vision alignment stage; the default Stage-2 inference script does not require raw EEG input.

`embeddings/rellisur_test_aligned_cognition_embeddings.pt` contains pre-computed aligned cognition embeddings for the RELLISUR test set. It is not a model checkpoint; it stores the Stage-1 output features that have already been computed for the test images. The file contains a `features` tensor with shape `[425, 1024]` and the corresponding image `paths` list. It is provided for reproducibility checks, analysis, and direct inspection of the cognition-aligned representations.

## Checkpoints

Download the large model and metric files from the anonymous Google Drive folder:

```text
https://drive.google.com/drive/folders/1pHCqSFy0fuOrhTXwL-WjkBkzc4sy1eV6?usp=drive_link
```

The Google Drive folder contains:

```text
checkpoints/
musiq_koniq_ckpt-e95806b9.pth
```

Additional EEG-related released files, if provided separately, should be placed according to the paths documented in the Stage-1 EEG/image alignment section above.

After downloading, place them under the repository root so that the structure becomes:

```text
checkpoints/stage1/image_40.pth
checkpoints/stage1/eeg_40.pth
checkpoints/stage2/model2.pkl
musiq_koniq_ckpt-e95806b9.pth
```

The following smaller files are already included in the code package:

```text
assets/mm-realsr/de_net.pth
niqe_modelparameters.mat
```

Download SD-Turbo from HuggingFace:

```text
https://huggingface.co/stabilityai/sd-turbo
```

Place the downloaded SD-Turbo directory at:

```text
pretrained/sd-turbo/
```

| File | Usage |
| --- | --- |
| `checkpoints/stage1/image_40.pth` | Stage-1 image encoder / alignment projector used to compute aligned cognition embeddings online |
| `checkpoints/stage1/eeg_40.pth` | Stage-1 EEG encoder checkpoint from contrastive cognition-vision alignment training; included for Stage-1 reproducibility |
| `checkpoints/stage2/model2.pkl` | CVSF restoration checkpoint |
| `assets/mm-realsr/de_net.pth` | degradation estimator |
| `pretrained/sd-turbo/` | SD-Turbo diffusion backbone |
| `musiq_koniq_ckpt-e95806b9.pth` | MUSIQ metric |
| `niqe_modelparameters.mat` | NIQE metric |
| `embeddings/rellisur_test_aligned_cognition_embeddings.pt` | pre-computed Stage-1 aligned cognition embeddings for the RELLISUR test set; feature file, not a model checkpoint |

`pretrained/sd-turbo/` should contain `model_index.json`, `scheduler/`, `text_encoder/`, `tokenizer/`, `unet/`, and `vae/`.

## Default Settings

| Setting | Value |
| --- | --- |
| task | RELLISUR restoration, X2 example |
| config | `configs/inference_x2.yaml` |
| batch size | 1 |
| tile size | 512 |
| tile overlap | 64 |
| output dir | `outputs/x2_test` |
| GPU | `--gpu_ids="0,"` |

To use another GPU, edit `inference.sh`, for example:

```bash
--gpu_ids="1,"
```

## Inference

```bash
sh inference.sh
```

Outputs:

```text
outputs/x2_test/pred
outputs/x2_test/gt
```

## Evaluation

```bash
sh evaluate.sh
```

Metrics:

```text
outputs/x2_test/metrics_all.txt
```

The evaluation wrapper computes metrics over the whole folder and does not split results by degradation-level suffixes.

## Training Templates

Stage-1 alignment:

```bash
sh scripts/train_stage1.sh
```

Stage-2 CVSF training:

```bash
sh scripts/train_stage2.sh
```

These scripts are launch templates. Full training requires the datasets and paths described above.

## Script Map

| Paper item | Script / file |
| --- | --- |
| Table 3 (x2 main results) | `inference.sh` with `configs/inference_x2.yaml` -> `evaluate.sh` |
| Table 3 (x4 main results) | `inference.sh` with `configs/inference_x4.yaml` -> `evaluate.sh` |
| Table 4 (cross-exposure, -3/-4/-5EV) | `inference.sh` with per-EV meta files -> `evaluate.sh` |
| Table 5 (prompt-based comparison) | `logs/all_compare_methods.log` |
| Table 7 (efficiency) | `configs/efficiency.yaml` |
| Figures 3-4 (visual comparison) | `outputs/x2_test/pred`, `outputs/x4_test/pred` |
| Stage-1 alignment training | `scripts/train_stage1.sh`, `configs/stage1_alignment.yaml` |
| Stage-2 CVSF training | `scripts/train_stage2.sh`, `configs/stage2_cvsf.yaml` |
| Pre-computed aligned cognition embeddings | `embeddings/rellisur_test_aligned_cognition_embeddings.pt` |
| Environment | `environment.yml`, `requirements.txt` |

## Baseline Reproduction

All baselines are evaluated using their respective official codebases. Inference logs for every method are provided in `logs/all_compare_methods.log`.

- Retrained baselines (NAFNet, HAT, DAT, CATANet, BSRGAN, CALGAN, SeeSR, OSEDiff, UPSR): trained from scratch on the RELLISUR training split using each method's official code and recommended hyperparameters.
- PromptIR: fine-tuned on RELLISUR from official pre-trained weights (batch=4, crop=256, lr=1e-5, 10,000 steps, bicubic pre-upsampling).
- InstructIR, DATPRL-IR: zero-shot with official multi-degradation checkpoints and bicubic pre-upsampling.
- 4KAgent, 4KAgent (Brightening), AgenticIR: zero-shot with official code and default configuration.

Detailed per-method configurations are listed in Appendix A.4 of the paper.

## Compared Methods in `logs/all_compare_methods.log`

```text
NAFNet
HAT
DAT
CATANet
BSRGAN
CALGAN
SeeSR
UPSR
OSEDiff
PromptIR
DATPRL-IR
InstructIR
4KAgent
4KAgent (Brightening)
AgenticIR
CVSF
```

## Repository Checklist

| Item | Files |
| --- | --- |
| Framework implementation | `src/cvsf.py`, `src/inference_cvsf.py`, `src/de_net.py`, `src/model.py`, `src/net_utils_ablation.py`, `src/my_utils/stage1_img_encoder.py`, `src/my_utils/stable_unet_adapter.py` |
| Alignment projector / Stage-1 encoder | `stage1/alignment_train_v3.py`, `stage1/model/ImageEncoder.py`, `stage1/model/alignment/SCAN.py`, `src/my_utils/stage1_img_encoder.py` |
| CMC-Attention / CDA-LoRA / diffusion model | `src/cvsf.py`, `src/model.py`, `src/my_utils/stable_unet_adapter.py` |
| Degradation estimator | `src/de_net.py`, `assets/mm-realsr/de_net.pth` |
| Stage-1 / Stage-2 training templates | `scripts/train_stage1.sh`, `scripts/train_stage2.sh`, `stage1/`, `src/train_cvsf.py` |
| Inference and evaluation scripts | `inference.sh`, `evaluate.sh`, `scripts/evaluate_img_new.py`, `scripts/evaluate_whole_folder.py` |
| YAML configuration files | `configs/inference_x2.yaml`, `configs/stage1_alignment.yaml`, `configs/stage2_cvsf.yaml`, `configs/efficiency.yaml` |
| Pretrained checkpoints | `checkpoints/stage1/image_40.pth`, `checkpoints/stage1/eeg_40.pth`, `checkpoints/stage2/model2.pkl`, `assets/mm-realsr/de_net.pth` |
| Pre-computed aligned cognition embeddings | `embeddings/rellisur_test_aligned_cognition_embeddings.pt` |
| Inference logs for compared methods | `logs/all_compare_methods.log` |
| Environment setup | `environment.yml`, `requirements.txt` |

## Notes

- This package is for inference and evaluation, with training launch templates.
- RELLISUR images and raw EEG data are not redistributed.
- SD-Turbo should be downloaded separately and placed in `pretrained/sd-turbo/`.
- If checkpoint names are changed, update the corresponding paths in `inference.sh`.
