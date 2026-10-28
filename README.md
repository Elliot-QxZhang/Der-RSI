# Derm-RSI-SAM

Recursive self-improvement (RSI) of SAM prompting policies for dermoscopic skin lesion segmentation.

SAM weights stay frozen. What improves over generations is everything around SAM: how the first prompt is generated, which corrective prompts and rules are applied, and the improver that decides where to look for the next improvement. The search runs offline on a development split, where ground truth is used to analyse failures. At inference time the learned policy is frozen and works from the image and SAM's own predictions only.

## Method

1. **Coarse localiser.** A U-Net (ResNet34, 512 px) gives a coarse mask and box for each image.
2. **Stage A, prompt adaptation** (`scripts/train_sam.py`). MedSAM ViT-B, fully frozen, plus learned prompts: 8 context tokens, a lesion token and a dense prompt bias (1.05M parameters), trained with a multi-round prompt loop. An alternative backbone is SAM 2.1 Hiera-L with LoRA r16.
3. **Stage B, RSI policy search** (`scripts/rsi_search.py`). An archive-based search over (policy, improver) pairs. Each generation runs a parent policy with SAM on dev-A, analyses dermoscopic failure modes, and lets the parent's improver propose child policies, using rule edits, mutations and optionally a local Qwen3-VL-8B-Instruct proposer. A child is kept only if it improves dev-A without degrading the held-out dev-B. The improver is credited for the gains it produces and is itself inherited and mutated.
4. **Inference** (`scripts/infer_policy.py`, `scripts/predict.py`). Coarse localiser, then DermSAM, then the frozen best policy, giving binary masks at the original resolution.

## Repository layout

```text
dermrsi/
  data.py          dataset splits and cached arrays
  metrics.py       Dice / IoU / boundary F1 / surface distances
  evaluate.py      scoring at original ground-truth resolution
  sam_model.py     DermSAM: SAM 2.1 or MedSAM wrapper with LoRA / prompt learning
  sam_infer.py     multi-round SAM inference used during training
  prompts.py       boxes, points and mask prompts
  baselines.py     U-Net coarse localiser (and other segmentation models)
  rsi/
    policy.py      policy representation and executor
    descriptors.py GT-free dermoscopic descriptors
    failure.py     failure analysis
    improver.py    improver (operators, credit assignment, meta-mutation)
    llm.py         Qwen3-VL-8B policy proposer
    fitness.py     fitness on dev-A / dev-B
scripts/
  prepare_data.py  build resized caches and splits
  train_sam.py     Stage A
  rsi_search.py    Stage B
  infer_policy.py  apply a frozen policy to a split and score it
  predict.py       end-to-end inference on arbitrary images
```

## Requirements

Tested with Python 3.10, PyTorch 2.4.1 (CUDA 12.4), transformers 4.57 (for `Sam2Model` and Qwen3-VL), segmentation-models-pytorch 0.4.0, timm 0.9.16, numpy, scipy, opencv-python and pillow.

```bash
conda create -n Derm-RSI python=3.10
conda activate Derm-RSI
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.57" segmentation-models-pytorch==0.4.0 timm==0.9.16 \
            numpy scipy opencv-python pillow
```

SAM 2.1 weights are downloaded from Hugging Face (`facebook/sam2.1-hiera-large`). MedSAM weights come from `flaviagiammarino/medsam-vit-base` and are expected as a local safetensors copy in `checkpoints/medsam-vit-base-fg/`.

## Data

Datasets are not included. Download them and extract to `data/raw/`:

```text
data/raw/
  ISIC2018_Task1-2_{Training,Validation,Test}_Input/
  ISIC2018_Task1_{Training,Validation,Test}_GroundTruth/
  PH2Dataset/
  ISIC-2017_Test_v2_Data/                # optional external split
  ISIC-2017_Test_v2_Part1_GroundTruth/
```

`python scripts/prepare_data.py` builds the caches and the splits:

| split | images | use |
| --- | --- | --- |
| train | 2294 | ISIC 2018 training set minus a fixed 300-image hold-out |
| dev | 400 | the hold-out plus the official validation set; checkpoint selection and RSI |
| test | 1000 | official ISIC 2018 Task 1 test set; reporting only |
| ph2 | 200 | external test |

## Usage

```bash
# Stage A: MedSAM prompt learning (writes runs/sam/medsam_prompt8_e20/)
torchrun --nproc_per_node 1 scripts/train_sam.py --variant medsam --tune prompt --n_ctx 8 --epochs 20 \
         --bs 8 --lr 2e-3 --wd 0 --lowres_p 0.3 --eval_every 2

# Stage B: RSI policy search (the LLM proposer needs a second GPU)
python -c "import json; from dermrsi.rsi.policy import default_policy; \
json.dump(default_policy(0, init='coarse'), open('root_coarse_policy.json', 'w'))"
python scripts/rsi_search.py --variant medsam --ckpt runs/sam/medsam_prompt8_e20/best.pt \
       --init_policy root_coarse_policy.json --gens 60 --llm 1 --llm_device cuda:1 --out runs/rsi/medsamP_llm_s0

# Apply the frozen policy and score dev / test / PH2 (writes runs/final/<name>/)
python scripts/infer_policy.py --variant medsam --ckpt runs/sam/medsam_prompt8_e20/best.pt \
       --policy runs/rsi/medsamP_llm_s0/best_policy.json --splits dev test ph2 --name final

# Inference on arbitrary images (writes <name>_mask.png)
python scripts/predict.py --images 'some/dir/*.jpg' --out preds/ \
       --sam_ckpt <sam.pt> --unet_ckpt <unet.pt> --policy <policy.json>
```

Notes:

- `rsi_search.py` and `infer_policy.py` read coarse-localiser predictions from `runs/baselines/unet_s512/pred_<split>.npy` (uint8 probabilities from the U-Net in `dermrsi/baselines.py`, `build("unet")`). The training script for this U-Net is not part of this repository.
- Without `--sam_ckpt`, `--unet_ckpt` and `--policy`, `predict.py` looks for the released files in `checkpoints/final/`. Checkpoints are not included in this repository.

## Results

ISIC 2018 Task 1 official test set (n = 1000). Per-image means at original resolution. Boundary F1 uses a tolerance of 0.8% of the image diagonal.

| method | Dice | IoU | BF1 |
| --- | --- | --- | --- |
| U-Net (ResNet34, 512 px) | 89.14 | 81.79 | 45.33 |
| MedSAM, zero-shot, automatic box | 90.06 | 83.32 | 50.52 |
| Derm-RSI-SAM, SAM 2.1-L backbone | 90.58 | 84.02 | 52.12 |
| Derm-RSI-SAM, MedSAM backbone (final) | 91.48 | 85.34 | 55.83 |

On PH² (n = 200, external) the final model reaches Dice 93.17 / IoU 87.79 / BF1 56.73. The final policy (the LLM-assisted search with seed 0) was chosen after test results were available; a second seed of the same search gives test Dice 91.28.
