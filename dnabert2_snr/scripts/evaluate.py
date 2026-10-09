#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import TFBSStage2Dataset, Stage2Collator
from model import Stage2TFBSModel

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# -------------------------
# basic utils
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_int_list(s: str) -> List[int]:
    s = str(s).strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_str_list(s: str) -> List[str]:
    s = str(s).strip()
    if s == "":
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_tf_names(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def append_jsonl(obj: Any, path: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def infer_tf_names_from_batch(batch: Dict[str, Any], batch_size: int) -> List[str]:
    candidate_keys = ["tf_names", "tf_name", "tfs", "tf"]

    for k in candidate_keys:
        if k not in batch:
            continue

        v = batch[k]

        if isinstance(v, list):
            return [str(x) for x in v]

        if isinstance(v, tuple):
            return [str(x) for x in v]

        if torch.is_tensor(v):
            v_cpu = v.detach().cpu().tolist()
            return [str(x) for x in v_cpu]

        if isinstance(v, str):
            return [v] * batch_size

    return ["UNKNOWN"] * batch_size

def create_length_collator(tokenizer, dna_length: int, tokenizer_max_length: Optional[int] = None):
    base_collator = Stage2Collator(
        tokenizer=tokenizer,
        dna_length=dna_length,
        tokenizer_max_length=tokenizer_max_length,
        debug=False,
    )

    def collate_fn(features):
        batch = base_collator(features)
        batch["dna_length"] = int(dna_length)
        return batch

    return collate_fn

def build_forward_kwargs(batch: Dict[str, Any], dna_length: int) -> Dict[str, Any]:
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "offset_mapping": batch["offset_mapping"],
        "tf_ppm": batch["tf_ppm"],
        "tf_ppm_mask": batch["tf_ppm_mask"],
        "seq_labels": batch.get("seq_labels", None),
        "token_labels": batch.get("token_labels", None),
        "token_valid_mask": batch.get("token_valid_mask", None),
        "nt_labels": batch.get("nt_labels", None),
        "tf_ppm_lengths": batch.get("tf_ppm_lengths", None),
        "dna_length": int(dna_length),
    }

    if "token_type_ids" in batch:
        kwargs["token_type_ids"] = batch["token_type_ids"]

    if "sequences" in batch:
        kwargs["sequences"] = batch["sequences"]

    return kwargs

def build_loader_for_length(
    data_root: str,
    ppm_root: str,
    split: str,
    length: int,
    tokenizer,
    batch_size: int,
    num_workers: int,
    tf_names: Optional[List[str]] = None,
    max_samples_per_tf: Optional[int] = None,
    tokenizer_max_length: Optional[int] = None,
    verbose: bool = True,
):
    dataset = TFBSStage2Dataset(
        data_root=data_root,
        ppm_root=ppm_root,
        split=split,
        length=length,
        tf_names=tf_names,
        max_samples_per_tf=max_samples_per_tf,
        require_ppm=True,
        uppercase=True,
        verbose=verbose,
    )

    collate_fn = create_length_collator(
        tokenizer=tokenizer,
        dna_length=length,
        tokenizer_max_length=tokenizer_max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )

    return dataset, loader

# -------------------------
# metric store
# -------------------------

def new_store():
    return {
        "seq_true": [],
        "seq_score": [],
        "tok_true": [],
        "tok_score": [],
        "nt_true": [],
        "nt_score": [],
    }

def merge_store(dst, src):
    for k in dst.keys():
        dst[k].extend(src[k])

def add_seq_to_store(store, y_true, y_score):
    store["seq_true"].extend(np.asarray(y_true).reshape(-1).tolist())
    store["seq_score"].extend(np.asarray(y_score).reshape(-1).tolist())

def add_tok_to_store(store, y_true, y_score):
    store["tok_true"].extend(np.asarray(y_true).reshape(-1).tolist())
    store["tok_score"].extend(np.asarray(y_score).reshape(-1).tolist())

def add_nt_to_store(store, y_true, y_score):
    store["nt_true"].extend(np.asarray(y_true).reshape(-1).tolist())
    store["nt_score"].extend(np.asarray(y_score).reshape(-1).tolist())

# -------------------------
# metrics
# -------------------------

def safe_auc_metrics(y_true, y_score, prefix: str, label_threshold: float = 0.5):
    out = {
        f"{prefix}_auroc": None,
        f"{prefix}_auprc": None,
    }

    if not SKLEARN_AVAILABLE:
        return out

    if len(y_true) == 0:
        return out

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    finite = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]

    if y_true.shape[0] == 0:
        return out

    y_bin = (y_true >= label_threshold).astype(np.int64)

    if len(np.unique(y_bin)) < 2:
        return out

    try:
        out[f"{prefix}_auroc"] = float(roc_auc_score(y_bin, y_score))
    except Exception:
        out[f"{prefix}_auroc"] = None

    try:
        out[f"{prefix}_auprc"] = float(average_precision_score(y_bin, y_score))
    except Exception:
        out[f"{prefix}_auprc"] = None

    return out

def safe_threshold_metrics(
    y_true,
    y_score,
    prefix: str,
    threshold: float = 0.5,
    label_threshold: float = 0.5,
):
    out = {
        f"{prefix}_precision": None,
        f"{prefix}_recall": None,
        f"{prefix}_f1": None,
        f"{prefix}_mcc": None,
        f"{prefix}_tp": 0,
        f"{prefix}_fp": 0,
        f"{prefix}_tn": 0,
        f"{prefix}_fn": 0,
        f"{prefix}_pos_rate": None,
        f"{prefix}_pred_pos_rate": None,
    }

    if len(y_true) == 0:
        return out

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)

    finite = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]

    if y_true.shape[0] == 0:
        return out

    y_bin = (y_true >= label_threshold).astype(np.int64)
    y_pred = (y_score >= threshold).astype(np.int64)

    tp = int(np.sum((y_bin == 1) & (y_pred == 1)))
    fp = int(np.sum((y_bin == 0) & (y_pred == 1)))
    tn = int(np.sum((y_bin == 0) & (y_pred == 0)))
    fn = int(np.sum((y_bin == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    denom = math.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    )
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    out[f"{prefix}_precision"] = float(precision)
    out[f"{prefix}_recall"] = float(recall)
    out[f"{prefix}_f1"] = float(f1)
    out[f"{prefix}_mcc"] = float(mcc)
    out[f"{prefix}_tp"] = tp
    out[f"{prefix}_fp"] = fp
    out[f"{prefix}_tn"] = tn
    out[f"{prefix}_fn"] = fn
    out[f"{prefix}_pos_rate"] = float(np.mean(y_bin))
    out[f"{prefix}_pred_pos_rate"] = float(np.mean(y_pred))

    return out

def compute_all_metrics(
    store,
    seq_threshold: float = 0.5,
    token_threshold: float = 0.5,
    nt_threshold: float = 0.5,
    seq_label_threshold: float = 0.5,
    token_label_threshold: float = 0.5,
    nt_label_threshold: float = 0.5,
):
    metrics = {}

    metrics.update(safe_auc_metrics(
        store["seq_true"],
        store["seq_score"],
        prefix="seq",
        label_threshold=seq_label_threshold,
    ))
    metrics.update(safe_auc_metrics(
        store["tok_true"],
        store["tok_score"],
        prefix="tok",
        label_threshold=token_label_threshold,
    ))
    metrics.update(safe_auc_metrics(
        store["nt_true"],
        store["nt_score"],
        prefix="nt",
        label_threshold=nt_label_threshold,
    ))

    metrics.update(safe_threshold_metrics(
        store["seq_true"],
        store["seq_score"],
        prefix="seq",
        threshold=seq_threshold,
        label_threshold=seq_label_threshold,
    ))
    metrics.update(safe_threshold_metrics(
        store["tok_true"],
        store["tok_score"],
        prefix="tok",
        threshold=token_threshold,
        label_threshold=token_label_threshold,
    ))
    metrics.update(safe_threshold_metrics(
        store["nt_true"],
        store["nt_score"],
        prefix="nt",
        threshold=nt_threshold,
        label_threshold=nt_label_threshold,
    ))

    metrics["num_seq_points"] = int(len(store["seq_true"]))
    metrics["num_token_points"] = int(len(store["tok_true"]))
    metrics["num_nt_points"] = int(len(store["nt_true"]))

    return metrics

def add_loss(loss_sums: Dict[str, float], outputs: Dict[str, torch.Tensor]):
    keys = [
        "loss",
        "loss_seq",
        "loss_tok",
        "loss_nt",
        "loss_tok_bce",
        "loss_tok_focal",
        "loss_nt_focal",
        "loss_nt_dice",
    ]

    for k in keys:
        if k in outputs:
            loss_sums[k] = loss_sums.get(k, 0.0) + float(outputs[k].detach().cpu().item())

def average_losses(loss_sums: Dict[str, float], n_steps: int):
    if n_steps <= 0:
        return {}
    return {k: float(v / n_steps) for k, v in loss_sums.items()}

# -------------------------
# model loading
# -------------------------

def get_arg(ckpt_args: Dict[str, Any], cli_args, name: str, default=None):
    if hasattr(cli_args, name):
        v = getattr(cli_args, name)
        if v is not None:
            return v
    return ckpt_args.get(name, default)

def build_model_from_checkpoint_args(cli_args, ckpt_args: Dict[str, Any]):
    model = Stage2TFBSModel(
        model_name_or_path=cli_args.model_name_or_path,
        dna_length=get_arg(ckpt_args, cli_args, "model_default_length", 200),
        dropout=get_arg(ckpt_args, cli_args, "dropout", 0.1),
        freeze_backbone=False,
        disable_flash=cli_args.disable_flash,

        ppm_conv_dim=get_arg(ckpt_args, cli_args, "ppm_conv_dim", 128),
        ppm_kernel_size=get_arg(ckpt_args, cli_args, "ppm_kernel_size", 3),

        fusion_type=get_arg(ckpt_args, cli_args, "fusion_type", "cross_attn"),
        ppm_max_len=get_arg(ckpt_args, cli_args, "ppm_max_len", 128),
        ppm_use_pos_embedding=get_arg(ckpt_args, cli_args, "ppm_use_pos_embedding", True),
        cross_attn_layers=get_arg(ckpt_args, cli_args, "cross_attn_layers", 1),
        cross_attn_heads=get_arg(ckpt_args, cli_args, "cross_attn_heads", 8),
        cross_attn_dropout=get_arg(ckpt_args, cli_args, "cross_attn_dropout", None),
        cross_attn_ffn_dim=get_arg(ckpt_args, cli_args, "cross_attn_ffn_dim", None),
        cross_attn_use_ffn=get_arg(ckpt_args, cli_args, "cross_attn_use_ffn", True),

        loss_seq_weight=get_arg(ckpt_args, cli_args, "loss_seq_weight", 1.0),
        loss_tok_weight=get_arg(ckpt_args, cli_args, "loss_tok_weight", 1.0),
        loss_nt_weight=get_arg(ckpt_args, cli_args, "loss_nt_weight", 1.0),

        token_focal_gamma=get_arg(ckpt_args, cli_args, "token_focal_gamma", 0.0),
        token_focal_alpha=get_arg(ckpt_args, cli_args, "token_focal_alpha", None),

        nt_focal_gamma=get_arg(ckpt_args, cli_args, "nt_focal_gamma", 2.0),
        nt_focal_alpha=get_arg(ckpt_args, cli_args, "nt_focal_alpha", 0.25),
        nt_focal_weight=get_arg(ckpt_args, cli_args, "nt_focal_weight", 0.5),
        nt_dice_weight=get_arg(ckpt_args, cli_args, "nt_dice_weight", 0.5),
        dice_smooth=get_arg(ckpt_args, cli_args, "dice_smooth", 1.0),
        # nt refinement head
        # Important:
        # default is "expand_logits" for backward compatibility with old checkpoints.
        nt_head_type=get_arg(ckpt_args, cli_args, "nt_head_type", "expand_logits"),
        nt_refine_dim=get_arg(ckpt_args, cli_args, "nt_refine_dim", None),
        nt_refine_kernel_size=get_arg(ckpt_args, cli_args, "nt_refine_kernel_size", 5),
        nt_refine_num_layers=get_arg(ckpt_args, cli_args, "nt_refine_num_layers", 2),
    )

    return model

# -------------------------
# evaluation
# -------------------------

@torch.no_grad()
def evaluate_length_split(
    model,
    loader,
    length: int,
    split: str,
    device,
    args,
):
    model.eval()

    overall_store = new_store()
    per_tf_store: Dict[str, Dict[str, List[float]]] = {}

    loss_sums = {}
    n_steps = 0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=f"eval split={split} L={length}", dynamic_ncols=True)

    for batch in iterator:
        batch = move_batch_to_device(batch, device)
        fw = build_forward_kwargs(batch, dna_length=length)

        outputs = model(**fw)

        add_loss(loss_sums, outputs)
        n_steps += 1

        B = outputs["seq_logits"].shape[0]
        tf_names = infer_tf_names_from_batch(batch, B)

        # sequence
        seq_probs = torch.sigmoid(outputs["seq_logits"]).detach().float().cpu().numpy()
        seq_true = batch["seq_labels"].detach().float().cpu().numpy()

        add_seq_to_store(overall_store, seq_true, seq_probs)

        # token
        tok_probs = torch.sigmoid(outputs["token_logits"]).detach().float().cpu()
        tok_true = batch["token_labels"].detach().float().cpu()
        tok_mask = batch["token_valid_mask"].detach().bool().cpu()

        if tok_mask.any():
            add_tok_to_store(
                overall_store,
                tok_true[tok_mask].reshape(-1).numpy(),
                tok_probs[tok_mask].reshape(-1).numpy(),
            )

        # nt
        nt_probs = torch.sigmoid(outputs["nt_logits"]).detach().float().cpu()
        nt_true = batch["nt_labels"].detach().float().cpu()

        add_nt_to_store(
            overall_store,
            nt_true.reshape(-1).numpy(),
            nt_probs.reshape(-1).numpy(),
        )

        # per-TF stores
        for i in range(B):
            tf = tf_names[i]
            if tf not in per_tf_store:
                per_tf_store[tf] = new_store()

            # seq one point
            add_seq_to_store(
                per_tf_store[tf],
                [float(seq_true[i])],
                [float(seq_probs[i])],
            )

            # token valid points for sample i
            m_i = tok_mask[i]
            if m_i.any():
                add_tok_to_store(
                    per_tf_store[tf],
                    tok_true[i][m_i].reshape(-1).numpy(),
                    tok_probs[i][m_i].reshape(-1).numpy(),
                )

            # nt points for sample i
            add_nt_to_store(
                per_tf_store[tf],
                nt_true[i].reshape(-1).numpy(),
                nt_probs[i].reshape(-1).numpy(),
            )

    losses = average_losses(loss_sums, n_steps)

    overall_metrics = compute_all_metrics(
        overall_store,
        seq_threshold=args.seq_threshold,
        token_threshold=args.token_threshold,
        nt_threshold=args.nt_threshold,
        seq_label_threshold=args.seq_label_threshold,
        token_label_threshold=args.token_label_threshold,
        nt_label_threshold=args.nt_label_threshold,
    )

    per_tf_metrics = {}

    for tf, store in sorted(per_tf_store.items()):
        m = compute_all_metrics(
            store,
            seq_threshold=args.seq_threshold,
            token_threshold=args.token_threshold,
            nt_threshold=args.nt_threshold,
            seq_label_threshold=args.seq_label_threshold,
            token_label_threshold=args.token_label_threshold,
            nt_label_threshold=args.nt_label_threshold,
        )
        m["tf_name"] = tf
        m["length"] = int(length)
        m["split"] = split
        per_tf_metrics[tf] = m

    result = {
        "split": split,
        "length": int(length),
        "num_steps": n_steps,
        "losses": losses,
        "metrics": overall_metrics,
        "per_tf_metrics": per_tf_metrics,
        "store": overall_store,
    }

    return result

def main():
    parser = argparse.ArgumentParser()

    # data/model
    parser.add_argument("--data_root", type=str, required=True, default="/bed_result_split")
    parser.add_argument("--ppm_root", type=str, required=True, default="/PPM_data")
    parser.add_argument("--model_name_or_path", type=str, required=True, default="/model")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # eval plan
    parser.add_argument("--lengths", type=str, required=True)
    parser.add_argument("--splits", type=str, required=True)
    parser.add_argument("--tf_names", type=str, default="")
    parser.add_argument("--max_samples_per_tf", type=int, default=None)

    # loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--tokenizer_max_length", type=int, default=None)

    # thresholds
    parser.add_argument("--seq_threshold", type=float, default=0.5)
    parser.add_argument("--token_threshold", type=float, default=0.5)
    parser.add_argument("--nt_threshold", type=float, default=0.5)

    parser.add_argument("--seq_label_threshold", type=float, default=0.5)
    parser.add_argument("--token_label_threshold", type=float, default=0.5)
    parser.add_argument("--nt_label_threshold", type=float, default=0.5)

    # runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_flash", action="store_true")

    # optional overrides, normally loaded from checkpoint args
    parser.add_argument("--model_default_length", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--ppm_conv_dim", type=int, default=None)
    parser.add_argument("--ppm_kernel_size", type=int, default=None)
    parser.add_argument("--fusion_type", type=str, default=None)
    parser.add_argument("--ppm_max_len", type=int, default=None)
    parser.add_argument("--ppm_use_pos_embedding", type=bool, default=None)
    parser.add_argument("--cross_attn_layers", type=int, default=None)
    parser.add_argument("--cross_attn_heads", type=int, default=None)
    parser.add_argument("--cross_attn_dropout", type=float, default=None)
    parser.add_argument("--cross_attn_ffn_dim", type=int, default=None)
    parser.add_argument("--cross_attn_use_ffn", type=bool, default=None)

    parser.add_argument("--loss_seq_weight", type=float, default=None)
    parser.add_argument("--loss_tok_weight", type=float, default=None)
    parser.add_argument("--loss_nt_weight", type=float, default=None)
    parser.add_argument("--token_focal_gamma", type=float, default=None)
    parser.add_argument("--token_focal_alpha", type=float, default=None)
    parser.add_argument("--nt_focal_gamma", type=float, default=None)
    parser.add_argument("--nt_focal_alpha", type=float, default=None)
    parser.add_argument("--nt_focal_weight", type=float, default=None)
    parser.add_argument("--nt_dice_weight", type=float, default=None)
    parser.add_argument("--dice_smooth", type=float, default=None)
    # nt refinement head, normally loaded from checkpoint args
    parser.add_argument("--nt_head_type", type=str, default=None)
    parser.add_argument("--nt_refine_dim", type=int, default=None)
    parser.add_argument("--nt_refine_kernel_size", type=int, default=None)
    parser.add_argument("--nt_refine_num_layers", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    lengths = parse_int_list(args.lengths)
    splits = parse_str_list(args.splits)

    if len(lengths) != len(splits):
        raise ValueError(
            f"--lengths and --splits must have same length, got "
            f"{len(lengths)} vs {len(splits)}"
        )

    tf_names = parse_tf_names(args.tf_names)

    save_json(vars(args), os.path.join(args.output_dir, "eval_args.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("evaluate_stage2_multilen")
    print("lengths:", lengths)
    print("splits:", splits)
    print("checkpoint:", args.checkpoint_path)
    print("device:", device)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {})
    print("checkpoint nt_head_type:", ckpt_args.get("nt_head_type", "MISSING"))
    print("checkpoint nt_refine_dim:", ckpt_args.get("nt_refine_dim", "MISSING"))
    print("checkpoint nt_refine_kernel_size:", ckpt_args.get("nt_refine_kernel_size", "MISSING"))
    print("checkpoint nt_refine_num_layers:", ckpt_args.get("nt_refine_num_layers", "MISSING"))

    model = build_model_from_checkpoint_args(args, ckpt_args)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    all_results = {}
    pooled_store = new_store()
    pooled_loss_sums = {}
    pooled_steps = 0

    length_metrics_jsonl = os.path.join(args.output_dir, "length_metrics.jsonl")
    per_tf_jsonl = os.path.join(args.output_dir, "per_tf_length_metrics.jsonl")

    # clean old files
    for p in [length_metrics_jsonl, per_tf_jsonl]:
        if os.path.exists(p):
            os.remove(p)

    for length, split in zip(lengths, splits):
        _, loader = build_loader_for_length(
            data_root=args.data_root,
            ppm_root=args.ppm_root,
            split=split,
            length=length,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tf_names=tf_names,
            max_samples_per_tf=args.max_samples_per_tf,
            tokenizer_max_length=args.tokenizer_max_length,
            verbose=True,
        )

        result = evaluate_length_split(
            model=model,
            loader=loader,
            length=length,
            split=split,
            device=device,
            args=args,
        )

        key = f"{split}_{length}"

        compact_result = {
            "split": split,
            "length": length,
            "num_steps": result["num_steps"],
            "losses": result["losses"],
            "metrics": result["metrics"],
        }

        all_results[key] = compact_result

        append_jsonl(compact_result, length_metrics_jsonl)

        # per tf jsonl
        for tf, tf_metrics in result["per_tf_metrics"].items():
            append_jsonl(tf_metrics, per_tf_jsonl)

        # pooled over all requested length/split
        merge_store(pooled_store, result["store"])

        for k, v in result["losses"].items():
            pooled_loss_sums[k] = pooled_loss_sums.get(k, 0.0) + float(v) * result["num_steps"]
        pooled_steps += result["num_steps"]

    pooled_losses = {
        k: float(v / max(pooled_steps, 1))
        for k, v in pooled_loss_sums.items()
    }

    pooled_metrics = compute_all_metrics(
        pooled_store,
        seq_threshold=args.seq_threshold,
        token_threshold=args.token_threshold,
        nt_threshold=args.nt_threshold,
        seq_label_threshold=args.seq_label_threshold,
        token_label_threshold=args.token_label_threshold,
        nt_label_threshold=args.nt_label_threshold,
    )

    pooled_result = {
        "losses": pooled_losses,
        "metrics": pooled_metrics,
    }

    save_json(all_results, os.path.join(args.output_dir, "length_metrics.json"))
    save_json(pooled_result, os.path.join(args.output_dir, "pooled_metrics.json"))

    print("=" * 80)
    print("Pooled metrics:")
    print(json.dumps(pooled_result, indent=2, ensure_ascii=False))
    print("=" * 80)

    print("Saved:")
    print(" -", os.path.join(args.output_dir, "length_metrics.json"))
    print(" -", os.path.join(args.output_dir, "length_metrics.jsonl"))
    print(" -", os.path.join(args.output_dir, "per_tf_length_metrics.jsonl"))
    print(" -", os.path.join(args.output_dir, "pooled_metrics.json"))

if __name__ == "__main__":
    main()

