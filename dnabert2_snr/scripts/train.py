#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import argparse
import random
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

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
# utils
# -------------------------

def parse_int_list(s: str) -> List[int]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_tf_names(s: Optional[str]):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

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

        # runtime length
        "dna_length": int(dna_length),
    }

    if "token_type_ids" in batch:
        kwargs["token_type_ids"] = batch["token_type_ids"]

    # For future motif-consistency loss. Current model ignores it.
    if "sequences" in batch:
        kwargs["sequences"] = batch["sequences"]

    return kwargs

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
    shuffle: bool = False,
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
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )

    return dataset, loader

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
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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
    pred_store: Dict[str, List[float]],
    seq_threshold: float = 0.5,
    token_threshold: float = 0.5,
    nt_threshold: float = 0.5,
    seq_label_threshold: float = 0.5,
    token_label_threshold: float = 0.5,
    nt_label_threshold: float = 0.5,
):
    metrics = {}

    metrics.update(safe_auc_metrics(
        pred_store["seq_true"],
        pred_store["seq_score"],
        prefix="seq",
        label_threshold=seq_label_threshold,
    ))
    metrics.update(safe_auc_metrics(
        pred_store["tok_true"],
        pred_store["tok_score"],
        prefix="tok",
        label_threshold=token_label_threshold,
    ))
    metrics.update(safe_auc_metrics(
        pred_store["nt_true"],
        pred_store["nt_score"],
        prefix="nt",
        label_threshold=nt_label_threshold,
    ))

    metrics.update(safe_threshold_metrics(
        pred_store["seq_true"],
        pred_store["seq_score"],
        prefix="seq",
        threshold=seq_threshold,
        label_threshold=seq_label_threshold,
    ))
    metrics.update(safe_threshold_metrics(
        pred_store["tok_true"],
        pred_store["tok_score"],
        prefix="tok",
        threshold=token_threshold,
        label_threshold=token_label_threshold,
    ))
    metrics.update(safe_threshold_metrics(
        pred_store["nt_true"],
        pred_store["nt_score"],
        prefix="nt",
        threshold=nt_threshold,
        label_threshold=nt_label_threshold,
    ))

    return metrics

def new_pred_store():
    return {
        "seq_true": [],
        "seq_score": [],
        "tok_true": [],
        "tok_score": [],
        "nt_true": [],
        "nt_score": [],
    }

def merge_pred_store(target, source):
    for k in target.keys():
        target[k].extend(source[k])

# -------------------------
# train / eval
# -------------------------

def build_model(args):
    model = Stage2TFBSModel(
        model_name_or_path=args.model_name_or_path,
        dna_length=args.model_default_length,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        disable_flash=args.disable_flash,

        ppm_conv_dim=args.ppm_conv_dim,
        ppm_kernel_size=args.ppm_kernel_size,

        fusion_type=args.fusion_type,
        ppm_max_len=args.ppm_max_len,
        ppm_use_pos_embedding=args.ppm_use_pos_embedding,
        cross_attn_layers=args.cross_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_dropout=args.cross_attn_dropout,
        cross_attn_ffn_dim=args.cross_attn_ffn_dim,
        cross_attn_use_ffn=args.cross_attn_use_ffn,

        loss_seq_weight=args.loss_seq_weight,
        loss_tok_weight=args.loss_tok_weight,
        loss_nt_weight=args.loss_nt_weight,

        token_focal_gamma=args.token_focal_gamma,
        token_focal_alpha=args.token_focal_alpha,

        nt_focal_gamma=args.nt_focal_gamma,
        nt_focal_alpha=args.nt_focal_alpha,
        nt_focal_weight=args.nt_focal_weight,
        nt_dice_weight=args.nt_dice_weight,
        dice_smooth=args.dice_smooth,
        nt_head_type=args.nt_head_type,
        nt_refine_dim=args.nt_refine_dim,
        nt_refine_kernel_size=args.nt_refine_kernel_size,
        nt_refine_num_layers=args.nt_refine_num_layers,
    )
    return model

def average_loss_dict(loss_sums, n_steps):
    if n_steps <= 0:
        return {}
    return {k: float(v / n_steps) for k, v in loss_sums.items()}

def add_loss(loss_sums, outputs):
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

def train_one_epoch_multilen(
    model,
    train_loaders: Dict[int, DataLoader],
    optimizer,
    scheduler,
    device,
    epoch: int,
    args,
):
    model.train()

    length_items = sorted(train_loaders.items(), key=lambda x: x[0])
    iterators = {L: iter(loader) for L, loader in length_items}
    active_lengths = set(iterators.keys())

    total_batches = sum(len(loader) for _, loader in length_items)

    loss_sums = {}
    n_steps = 0

    last_milestone = 0  # integer 0..10 for 0%..100%

    optimizer.zero_grad(set_to_none=True)

    while len(active_lengths) > 0:
        for L, _ in length_items:
            if L not in active_lengths:
                continue

            try:
                batch = next(iterators[L])
            except StopIteration:
                active_lengths.remove(L)
                continue

            batch = move_batch_to_device(batch, device)
            fw = build_forward_kwargs(batch, dna_length=L)

            outputs = model(**fw)
            loss = outputs["loss"]

            loss.backward()

            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            add_loss(loss_sums, outputs)
            n_steps += 1

            # report progress every 10%
            current_milestone = int(n_steps * 100 / total_batches)
            if current_milestone > last_milestone:
                pct = n_steps / total_batches * 100
                print(f"train epoch {epoch}: {pct:.1f}% [{n_steps}/{total_batches}], L={L}, loss={float(loss.detach().cpu().item()):.4f}")
                last_milestone = current_milestone

    return average_loss_dict(loss_sums, n_steps), n_steps

@torch.no_grad()
def evaluate_one_length(
    model,
    loader: DataLoader,
    length: int,
    device,
    args,
):
    model.eval()

    pred_store = new_pred_store()
    loss_sums = {}
    n_steps = 0

    total_batches = len(loader)
    last_milestone = 0  # integer 0..10 for 0%..100%

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        fw = build_forward_kwargs(batch, dna_length=length)

        outputs = model(**fw)

        add_loss(loss_sums, outputs)
        n_steps += 1

        # report progress every 10%
        current_milestone = int(n_steps * 10 / total_batches)
        if current_milestone > last_milestone:
            pct = n_steps / total_batches * 100
            loss_str = f"loss={float(outputs['loss'].detach().cpu().item()):.4f}"
            print(f"eval L={length}: {pct:.1f}% [{n_steps}/{total_batches}], {loss_str}")
            last_milestone = current_milestone

        # seq
        seq_probs = torch.sigmoid(outputs["seq_logits"]).detach().float().cpu().reshape(-1).numpy()
        seq_true = batch["seq_labels"].detach().float().cpu().reshape(-1).numpy()
        pred_store["seq_score"].extend(seq_probs.tolist())
        pred_store["seq_true"].extend(seq_true.tolist())

        # token
        token_probs = torch.sigmoid(outputs["token_logits"]).detach().float().cpu()
        token_labels = batch["token_labels"].detach().float().cpu()
        token_mask = batch["token_valid_mask"].detach().bool().cpu()

        if token_mask.any():
            pred_store["tok_score"].extend(token_probs[token_mask].reshape(-1).numpy().tolist())
            pred_store["tok_true"].extend(token_labels[token_mask].reshape(-1).numpy().tolist())

        # nt
        nt_probs = torch.sigmoid(outputs["nt_logits"]).detach().float().cpu()
        nt_labels = batch["nt_labels"].detach().float().cpu()

        pred_store["nt_score"].extend(nt_probs.reshape(-1).numpy().tolist())
        pred_store["nt_true"].extend(nt_labels.reshape(-1).numpy().tolist())

    loss_avg = average_loss_dict(loss_sums, n_steps)

    metrics = compute_all_metrics(
        pred_store=pred_store,
        seq_threshold=args.seq_threshold,
        token_threshold=args.token_threshold,
        nt_threshold=args.nt_threshold,
        seq_label_threshold=args.seq_label_threshold,
        token_label_threshold=args.token_label_threshold,
        nt_label_threshold=args.nt_label_threshold,
    )

    return {
        "length": length,
        "num_steps": n_steps,
        "losses": loss_avg,
        "metrics": metrics,
        "pred_store": pred_store,
    }

@torch.no_grad()
def evaluate_multilen(
    model,
    val_loaders: Dict[int, DataLoader],
    device,
    args,
):
    per_length = {}
    pooled_store = new_pred_store()

    pooled_loss_sums = {}
    pooled_steps = 0

    for L in sorted(val_loaders.keys()):
        result = evaluate_one_length(
            model=model,
            loader=val_loaders[L],
            length=L,
            device=device,
            args=args,
        )

        per_length[str(L)] = {
            "length": L,
            "num_steps": result["num_steps"],
            "losses": result["losses"],
            "metrics": result["metrics"],
        }

        merge_pred_store(pooled_store, result["pred_store"])

        for k, v in result["losses"].items():
            pooled_loss_sums[k] = pooled_loss_sums.get(k, 0.0) + float(v) * result["num_steps"]
        pooled_steps += result["num_steps"]

    pooled_losses = {
        k: float(v / max(pooled_steps, 1))
        for k, v in pooled_loss_sums.items()
    }

    pooled_metrics = compute_all_metrics(
        pred_store=pooled_store,
        seq_threshold=args.seq_threshold,
        token_threshold=args.token_threshold,
        nt_threshold=args.nt_threshold,
        seq_label_threshold=args.seq_label_threshold,
        token_label_threshold=args.token_label_threshold,
        nt_label_threshold=args.nt_label_threshold,
    )

    return {
        "pooled": {
            "losses": pooled_losses,
            "metrics": pooled_metrics,
        },
        "per_length": per_length,
    }

def flatten_eval_result(eval_result: Dict[str, Any], prefix: str = "val"):
    row = {}

    # pooled losses
    for k, v in eval_result["pooled"]["losses"].items():
        row[f"{prefix}_{k}"] = v

    # pooled metrics
    for k, v in eval_result["pooled"]["metrics"].items():
        row[f"{prefix}_{k}"] = v

    # per-length summary, nested kept separately in json
    return row

def save_checkpoint(path, model, optimizer, scheduler, args, epoch, global_step, best_metric):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }
    torch.save(ckpt, path)

def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_root", type=str, required=True, default="/bed_result_split")
    parser.add_argument("--ppm_root", type=str, required=True, default="/PPM_data")
    parser.add_argument("--model_name_or_path", type=str, required=True, default="/model")
    parser.add_argument("--output_dir", type=str, required=True, default="/output")

    parser.add_argument("--train_lengths", type=str, default="100,150,200")
    parser.add_argument("--val_lengths", type=str, default="100,150,200")
    parser.add_argument("--split_train", type=str, default="train")
    parser.add_argument("--split_val", type=str, default="val")

    parser.add_argument("--tf_names", type=str, default="")
    parser.add_argument("--max_train_samples_per_tf", type=int, default=None)
    parser.add_argument("--max_val_samples_per_tf", type=int, default=None)
    parser.add_argument("--max_samples_per_tf", type=int, default=None)

    # loader
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tokenizer_max_length", type=int, default=None)

    # training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # model
    parser.add_argument("--model_default_length", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--disable_flash", action="store_true")

    parser.add_argument("--ppm_conv_dim", type=int, default=128)
    parser.add_argument("--ppm_kernel_size", type=int, default=3)

    parser.add_argument("--fusion_type", type=str, default="cross_attn")
    parser.add_argument("--ppm_max_len", type=int, default=128)
    parser.add_argument("--ppm_use_pos_embedding", action="store_true")
    parser.add_argument("--no_ppm_use_pos_embedding", dest="ppm_use_pos_embedding", action="store_false")
    parser.set_defaults(ppm_use_pos_embedding=True)

    parser.add_argument("--cross_attn_layers", type=int, default=1)
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--cross_attn_dropout", type=float, default=None)
    parser.add_argument("--cross_attn_ffn_dim", type=int, default=None)
    parser.add_argument("--cross_attn_use_ffn", action="store_true")
    parser.add_argument("--no_cross_attn_use_ffn", dest="cross_attn_use_ffn", action="store_false")
    parser.set_defaults(cross_attn_use_ffn=True)

    # loss
    parser.add_argument("--loss_seq_weight", type=float, default=0.1)
    parser.add_argument("--loss_tok_weight", type=float, default=0.2)
    parser.add_argument("--loss_nt_weight", type=float, default=0.7)

    parser.add_argument("--token_focal_gamma", type=float, default=2.0)
    parser.add_argument("--token_focal_alpha", type=float, default=0.25)

    parser.add_argument("--nt_focal_gamma", type=float, default=2.0)
    parser.add_argument("--nt_focal_alpha", type=float, default=0.25)
    parser.add_argument("--nt_focal_weight", type=float, default=0.5)
    parser.add_argument("--nt_dice_weight", type=float, default=0.5)
    parser.add_argument("--dice_smooth", type=float, default=1.0)

    # nt refinement head
    parser.add_argument("--nt_head_type", type=str, default="refine_conv") # expand_logits为关闭refine_conv
    parser.add_argument("--nt_refine_dim", type=int, default=None)
    parser.add_argument("--nt_refine_kernel_size", type=int, default=3)
    parser.add_argument("--nt_refine_num_layers", type=int, default=2)

    # thresholds
    parser.add_argument("--seq_threshold", type=float, default=0.5)
    parser.add_argument("--token_threshold", type=float, default=0.5)
    parser.add_argument("--nt_threshold", type=float, default=0.5)

    parser.add_argument("--seq_label_threshold", type=float, default=0.5)
    parser.add_argument("--token_label_threshold", type=float, default=0.5)
    parser.add_argument("--nt_label_threshold", type=float, default=0.5)

    # checkpoint
    parser.add_argument("--save_last", action="store_true")

    args = parser.parse_args()
    if args.max_samples_per_tf is not None:
        if args.max_train_samples_per_tf is None:
            args.max_train_samples_per_tf = args.max_samples_per_tf
        if args.max_val_samples_per_tf is None:
            args.max_val_samples_per_tf = args.max_samples_per_tf

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_lengths = parse_int_list(args.train_lengths)
    val_lengths = parse_int_list(args.val_lengths)

    if len(train_lengths) == 0:
        raise ValueError("--train_lengths is empty.")
    if len(val_lengths) == 0:
        raise ValueError("--val_lengths is empty.")

    tf_names = parse_tf_names(args.tf_names)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    save_json(vars(args), os.path.join(args.output_dir, "args.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Multi-length Stage2 training")
    print("train_lengths:", train_lengths)
    print("val_lengths:", val_lengths)
    print("tf_names:", tf_names if tf_names is not None else "ALL")
    print("device:", device)
    print("nt_head_type:", args.nt_head_type)
    print("nt_refine_dim:", args.nt_refine_dim)
    print("nt_refine_kernel_size:", args.nt_refine_kernel_size)
    print("nt_refine_num_layers:", args.nt_refine_num_layers)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    # -------------------------
    # dataloaders
    # -------------------------
    train_loaders = {}
    for L in train_lengths:
        _, loader = build_loader_for_length(
            data_root=args.data_root,
            ppm_root=args.ppm_root,
            split=args.split_train,
            length=L,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            tf_names=tf_names,
            max_samples_per_tf=args.max_train_samples_per_tf,
            tokenizer_max_length=args.tokenizer_max_length,
            shuffle=True,
            verbose=True,
        )
        train_loaders[L] = loader

    val_loaders = {}
    for L in val_lengths:
        _, loader = build_loader_for_length(
            data_root=args.data_root,
            ppm_root=args.ppm_root,
            split=args.split_val,
            length=L,
            tokenizer=tokenizer,
            batch_size=eval_batch_size,
            num_workers=args.num_workers,
            tf_names=tf_names,
            max_samples_per_tf=args.max_val_samples_per_tf,
            tokenizer_max_length=args.tokenizer_max_length,
            shuffle=False,
            verbose=True,
        )
        val_loaders[L] = loader

    # -------------------------
    # model
    # -------------------------
    model = build_model(args)
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln"]
    grouped_params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        grouped_params,
        lr=args.learning_rate,
    )

    total_train_steps = sum(len(loader) for loader in train_loaders.values()) * args.num_epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    print("total_train_steps:", total_train_steps)
    print("warmup_steps:", warmup_steps)

    best_metric = -1.0
    best_epoch = -1
    global_step = 0

    log_path = os.path.join(args.output_dir, "train_log.jsonl")

    # -------------------------
    # train loop
    # -------------------------
    for epoch in range(1, args.num_epochs + 1):
        start = time.time()

        train_losses, n_steps = train_one_epoch_multilen(
            model=model,
            train_loaders=train_loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            args=args,
        )
        global_step += n_steps

        eval_result = evaluate_multilen(
            model=model,
            val_loaders=val_loaders,
            device=device,
            args=args,
        )

        epoch_time = time.time() - start

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "epoch_time_sec": epoch_time,
        }

        # train losses
        for k, v in train_losses.items():
            row[f"train_{k}"] = v

        # pooled val losses / metrics
        flat_val = flatten_eval_result(eval_result, prefix="val")
        row.update(flat_val)

        # nested per-length
        row["per_length"] = eval_result["per_length"]

        # checkpoint selection by pooled val_nt_auprc
        current_metric = eval_result["pooled"]["metrics"].get("nt_auprc", None)
        if current_metric is None:
            current_metric = -1.0

        is_best = float(current_metric) > float(best_metric)

        if is_best:
            best_metric = float(current_metric)
            best_epoch = epoch

            save_checkpoint(
                path=os.path.join(args.output_dir, "checkpoint_best.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
            )

        if args.save_last:
            save_checkpoint(
                path=os.path.join(args.output_dir, "checkpoint_last.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
            )

        row["best_metric"] = best_metric
        row["best_epoch"] = best_epoch
        row["is_best"] = is_best

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        save_json(
            {
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "latest": row,
            },
            os.path.join(args.output_dir, "training_state.json"),
        )

        print("=" * 80)
        print(json.dumps(row, indent=2, ensure_ascii=False))
        print("=" * 80)

    print("Training finished.")
    print("Best pooled val nt_auprc:", best_metric)
    print("Best epoch:", best_epoch)

if __name__ == "__main__":
    main()
