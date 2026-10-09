# DNABERT2-SNR

**DNABERT2-SNR** is a deep-learning framework for **nucleotide-resolution transcription-factor binding-site (TFBS) prediction**. It conditions the [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2) genomic foundation model on a transcription factor's **position probability matrix (PPM)** through a **DNA→PPM cross-attention** module, and predicts binding at three granularities simultaneously — sequence-level, token-level, and nucleotide-level.

The model is trained in a multi-length, multi-TF setting (DNA inputs of 100 / 150 / 200 bp) and is shown to **generalize to unseen lengths** (evaluated at 250 bp). In head-to-head comparisons across dozens of transcription factors, DNABERT2-SNR systematically outperforms existing TFBS predictors, including **BertSNR**, **DeepSNR**, and **D-AEDNet**.

---

## Table of contents

- [Key features](#key-features)
- [Method overview](#method-overview)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

---

## Key features

- **Foundation-model backbone** — reuses the pre-trained DNABERT-2 encoder (with the optional flash-attention disabled for compatibility) and only fine-tunes lightweight, task-specific heads.
- **PPM conditioning via cross-attention** — a TF's PPM is encoded into position-wise embeddings (`Key`/`Value`), and DNA hidden states (`Query`) attend to them, letting the network learn *which* motif positions matter for a given TF.
- **Multi-granularity prediction** — three heads produce sequence-, BPE-token-, and nucleotide-level binding probabilities in one forward pass.
- **Nucleotide-refinement head** — a small convolutional head (`refine_conv`) maps token hidden states back to nucleotide resolution, or the `expand_logits` mode expands token logits directly via the tokenizer's `offset_mapping`.
- **Task-aware losses** — sequence BCE, token-level focal loss, and nucleotide-level **focal + Dice** loss with configurable weights.
- **Multi-length training** — trains on 100/150/200 bp simultaneously and generalizes to 250 bp at inference time.
- **Rich analysis tooling** — cross-attention heatmaps, backbone self-attention flow plots, and per-nucleotide summary figures in a publication-ready (`Nature`-style) format.

---

## Method overview

```
                    ┌─────────────────────────────────────────────┐
                    │                 TF PPM (4 × M)              │
                    │         A C G T position probability matrix │
                    └──────────────────┬──────────────────────────┘
                                       │  PPMPositionEncoder
                                       │  (Conv1d + pos-embedding)
                                       ▼
                        ppm_hidden  [B, M, H]   (Key / Value)
                                       │
 DNA sequence ──► DNABERT-2 backbone ──► dna_hidden [B, T, H]  (Query)
   (100–250 bp)                         │
                                       ▼
                  DNA→PPM Cross-Attention (fusion_type="cross_attn")
                                       │
                        conditioned_hidden [B, T, H]
                          ┌────────────┼────────────────┐
                          ▼            ▼                ▼
                    seq_classifier  token_classifier   nt_refine_head
                     [B] (CLS)      [B, T]             [B, L]
                          │            │                │
                    sequence-level  token-level      nucleotide-level
                        binding       binding           binding
```

Two fusion modes are supported:

| `fusion_type` | Behavior |
|---------------|----------|
| `cross_attn`  | DNA hidden states attend to PPM position embeddings via cross-attention (default). |
| `add`         | A global PPM embedding is broadcast-added to the DNA hidden states. |

---

## Repository structure

```
dnbert2_snr/
├── scripts/
│   ├── model.py                                # DNABERT2SNRStage2 (Stage2TFBSModel) + PPM encoders + cross-attention fusion + losses
│   ├── dataset.py                              # TFBSStage2Dataset + Stage2Collator (PPM loading, label parsing)
│   ├── utils.py                                # DNABERT-2 backbone / tokenizer loading
│   ├── train.py                                # multi-length training loop
│   └── evaluate.py                             # evaluation (per-TF + pooled metrics)
```

> **Note.** `output/` contains trained checkpoints and generated figures. It is large and should generally be excluded from a public repository — see the suggested [`.gitignore`](#gitignore) below.

---

## Requirements

- Python 3.10+
- PyTorch ≥ 1.13
- Transformers ≥ 4.30
- NumPy, Pandas, Scikit-learn
- A pre-trained DNABERT-2 checkpoint (`model_name_or_path`)

Install with:

```bash
pip install torch transformers numpy pandas scikit-learn matplotlib seaborn scipy tqdm
```

---

## Installation

```bash
git clone https://github.com/<your-org>/dnbert2_snr.git
cd dnbert2_snr/scripts
```

Make sure the DNABERT-2 model directory (containing `pytorch_model.bin`, `config.json`, and tokenizer files) is available locally, then point `--model_name_or_path` to it.

---

## Data preparation

### DNA sequences and labels

The dataset is organized per length, per TF, per split:

```
data_root/
└── {length}/            # e.g. 100, 150, 200, 250
    └── {TF_name}/       # e.g. ASCL1, MITF, BACH1, ...
        └── {split}/     # train / val / unseen_test
            ├── sequence.txt    # one DNA sequence per line (length == {length})
            ├── label.txt       # one per-nucleotide binary label per line
            └── meta.tsv        # optional metadata (sample_id, ...)
```

`label.txt` may be written in compact (`000111000`), space-separated, or tab-separated form; each entry is a `0`/`1` flag marking whether that nucleotide belongs to a TFBS.

### TF position probability matrices (PPM)

Each TF needs a PPM file at `ppm_root/{TF_name}.txt`, with one row per base and space-separated probabilities summing to 1 per column:

```
A 0.10 0.95 0.05 ...
C 0.40 0.01 0.05 ...
G 0.35 0.02 0.85 ...
T 0.15 0.02 0.05 ...
```

---

## Training

Train a multi-length model with cross-attention fusion:

```bash
cd scripts

python train.py \
  --data_root /path/to/bed_split_result \
  --ppm_root  /path/to/PPM_data \
  --model_name_or_path /path/to/dnabert2 \
  --output_dir /path/to/output/All_TF \
  --train_lengths 100,150,200 \
  --val_lengths   100,150,200 \
  --split_train train \
  --split_val   val \
  --batch_size 32 \
  --eval_batch_size 32 \
  --num_workers 12 \
  --max_train_samples_per_tf 15000 \
  --max_val_samples_per_tf   1875 \
  --num_epochs 1 \
  --learning_rate 2e-5 \
  --fusion_type cross_attn \
  --nt_head_type refine_conv \
  --disable_flash \
  --save_last \
  --tf_names BCL11A,ASCL1,BACH1,BACH2,BATF3
```

Key hyperparameters:

| Argument | Default | Description |
|----------|---------|-------------|
| `--fusion_type` | `cross_attn` | `cross_attn` or `add` |
| `--nt_head_type` | `refine_conv` | `refine_conv` or `expand_logits` (disable refinement) |
| `--cross_attn_layers` / `--cross_attn_heads` | `1` / `8` | cross-attention depth / heads |
| `--loss_seq_weight` / `--loss_tok_weight` / `--loss_nt_weight` | `0.1` / `0.2` / `0.7` | multi-task loss weights |
| `--nt_focal_gamma` / `--nt_focal_alpha` | `2.0` / `0.25` | nucleotide focal-loss parameters |
| `--nt_focal_weight` / `--nt_dice_weight` | `0.5` / `0.5` | focal vs. Dice blending |
| `--freeze_backbone` | off | freeze the DNABERT-2 encoder |
| `--disable_flash` | off | disable flash attention for compatibility |

The best checkpoint is selected by pooled validation `nt_auprc` and written to `output_dir/checkpoint_best.pt`.

---

## Evaluation

Evaluate a trained checkpoint across lengths/splits (including an unseen 250 bp test split):

```bash
python evaluate.py \
  --data_root /path/to/bed_split_result \
  --ppm_root  /path/to/PPM_data \
  --model_name_or_path /path/to/dnabert2 \
  --checkpoint_path /path/to/checkpoint_best.pt \
  --output_dir /path/to/eval_per_tf \
  --lengths 100,150,200,250 \
  --splits  val,val,val,unseen_test \
  --tf_names ASCL1 \
  --disable_flash
```

The script reports pooled and per-length metrics, and writes:

- `length_metrics.json` / `length_metrics.jsonl` — per-(split, length) results
- `per_tf_length_metrics.jsonl` — per-TF metrics (used by `extract_dnabert2snr_metrics.py`)
- `pooled_metrics.json` — pooled over all requested lengths/splits

Reported metrics include **AUROC, AUPRC, precision, recall, F1, MCC** at sequence, token, and nucleotide level.

---
## Results

Per-TF metrics are stored as tab-separated tables (`TF_name, Acc, Pre, Rec, F1-S, AUC, AUPR, MCC`). Example excerpt (DNABERT2-SNR, all-length training, 100-trial):

| TF_name | Acc   | Pre   | Rec   | F1-S  | AUC   | AUPR  | MCC   |
|---------|-------|-------|-------|-------|-------|-------|-------|
| ASCL1   | 0.971 | 0.827 | 0.813 | 0.820 | 0.975 | 0.880 | 0.804 |
| ATF2    | 0.992 | 0.935 | 0.907 | 0.921 | 0.991 | 0.955 | 0.916 |
| ATF4    | 0.993 | 0.943 | 0.919 | 0.931 | 0.993 | 0.966 | 0.927 |
| BACH1   | 0.980 | 0.898 | 0.860 | 0.879 | 0.984 | 0.928 | 0.868 |


---

## Citation

If you use DNABERT2-SNR in your research, please cite the accompanying paper and the underlying models:

```bibtex
@article{dnabert2snr,
  title   = {STILL WORKING ON IT ...},
  author  = {},
  journal = {},
  year    = {}
}

@article{zhou2024dnabert2,
  title   = {DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genomes},
  author  = {Zhou, Zhihan and others},
  journal = {International Conference on Learning Representations (ICLR)},
  year    = {2024}
}
```

---

## License

This project is released for non-commercial research use. See `LICENSE` for details (if applicable).
