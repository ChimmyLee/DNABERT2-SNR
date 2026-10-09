#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

BASES = ["A", "C", "G", "T"]

def load_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def load_meta_tsv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [dict(row) for row in reader]

    return rows

def parse_nt_label_line(line: str, expected_len: int) -> np.ndarray:
    """
    支持三种 label.txt 格式：

    1. 紧凑格式:
        0001110000

    2. 空格分隔:
        0 0 0 1 1 1 0 0

    3. tab 分隔:
        0\t0\t1\t1\t0
    """
    s = line.strip()

    if len(s) == 0:
        raise ValueError("Empty label line.")

    if (" " not in s) and ("\t" not in s):
        if len(s) != expected_len:
            raise ValueError(
                f"Compact label length mismatch: got {len(s)}, expected {expected_len}"
            )
        arr = np.asarray([float(ch) for ch in s], dtype=np.float32)
    else:
        parts = s.replace("\t", " ").split()
        if len(parts) != expected_len:
            raise ValueError(
                f"Separated label length mismatch: got {len(parts)}, expected {expected_len}"
            )
        arr = np.asarray([float(x) for x in parts], dtype=np.float32)

    if not np.all((arr == 0.0) | (arr == 1.0)):
        raise ValueError("nt label must be binary 0/1.")

    return arr

def validate_sequences_and_labels(
    sequences: List[str],
    labels: List[str],
    expected_len: int,
    tf_name: str,
    split: str,
):
    if len(sequences) != len(labels):
        raise ValueError(
            f"[{tf_name}/{split}] sequence count != label count: "
            f"{len(sequences)} vs {len(labels)}"
        )

    for i, seq in enumerate(sequences):
        seq = seq.strip()
        if len(seq) != expected_len:
            raise ValueError(
                f"[{tf_name}/{split}] sequence length mismatch at idx={i}: "
                f"got {len(seq)}, expected {expected_len}"
            )

def load_ppm_file(path: str, check_col_sum: bool = True) -> np.ndarray:
    """
    读取已经生成好的 PPM 文件。

    输入格式：
        A 0.1 0.2 ...
        C 0.3 0.1 ...
        G 0.4 0.5 ...
        T 0.2 0.2 ...

    返回：
        ppm: np.ndarray, shape [4, M]
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PPM file not found: {path}")

    base_to_values: Dict[str, List[float]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            parts = s.split()
            base = parts[0].upper()

            if base not in BASES:
                raise ValueError(f"Invalid base row in PPM file {path}: {base}")

            values = [float(x) for x in parts[1:]]
            base_to_values[base] = values

    missing = [b for b in BASES if b not in base_to_values]
    if missing:
        raise ValueError(f"Missing base rows in PPM file {path}: {missing}")

    lengths = [len(base_to_values[b]) for b in BASES]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent PPM motif lengths in {path}: {lengths}")

    ppm = np.asarray([base_to_values[b] for b in BASES], dtype=np.float32)

    if check_col_sum:
        col_sums = ppm.sum(axis=0)
        max_err = float(np.max(np.abs(col_sums - 1.0)))
        if max_err > 1e-4:
            raise ValueError(
                f"PPM columns do not sum to 1 in {path}. max_err={max_err}"
            )

    return ppm

def nt_labels_to_token_labels(
    offset_mapping: List[Tuple[int, int]],
    nt_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 tokenizer 的 offset_mapping，把 nucleotide-level label
    映射成 token-level soft label。

    对每个 token：
        token_label = 该 token 覆盖的 nt_labels 平均值

    对 special token：
        offset 通常为 (0, 0)，token_valid_mask=False
    """
    token_labels = []
    token_valid_mask = []

    L = len(nt_labels)

    for start, end in offset_mapping:
        start = int(start)
        end = int(end)

        if end <= start:
            token_labels.append(0.0)
            token_valid_mask.append(False)
            continue

        # 防止 tokenizer offset 超出序列长度
        start = max(0, min(start, L))
        end = max(0, min(end, L))

        if end <= start:
            token_labels.append(0.0)
            token_valid_mask.append(False)
            continue

        span = nt_labels[start:end]
        token_labels.append(float(np.mean(span)))
        token_valid_mask.append(True)

    return (
        np.asarray(token_labels, dtype=np.float32),
        np.asarray(token_valid_mask, dtype=np.bool_),
    )

class TFBSStage2Dataset(Dataset):
    """
    PPM-conditioned multi-TF dataset.

    数据目录：
        data_root/{length}/{TF_name}/{split}/sequence.txt
        data_root/{length}/{TF_name}/{split}/label.txt
        data_root/{length}/{TF_name}/{split}/meta.tsv

    PPM 目录：
        ppm_root/{TF_name}.txt

    每个样本返回：
        sample_id
        tf_name
        sequence
        seq_label
        nt_labels
        tf_ppm
        meta
    """

    def __init__(
        self,
        data_root: str,
        ppm_root: str,
        split: str,
        length: int = 100,
        tf_names: Optional[List[str]] = None,
        max_samples_per_tf: Optional[int] = None,
        require_ppm: bool = True,
        uppercase: bool = True,
        verbose: bool = True,
    ):
        super().__init__()

        self.data_root = data_root
        self.ppm_root = ppm_root
        self.split = split
        self.length = int(length)
        self.require_ppm = require_ppm
        self.uppercase = uppercase
        self.verbose = verbose

        self.length_root = os.path.join(self.data_root, str(self.length))
        if not os.path.isdir(self.length_root):
            raise FileNotFoundError(f"Length root not found: {self.length_root}")

        available_tfs = sorted([
            x for x in os.listdir(self.length_root)
            if os.path.isdir(os.path.join(self.length_root, x))
        ])

        if tf_names is None:
            tf_names = available_tfs
        else:
            tf_names = list(tf_names)

        self.requested_tf_names = tf_names
        self.tf_names: List[str] = []
        self.tf_to_ppm: Dict[str, np.ndarray] = {}
        self.samples: List[Dict[str, Any]] = []
        self.skipped_tfs: List[Dict[str, str]] = []

        for tf_name in tf_names:
            tf_split_dir = os.path.join(self.length_root, tf_name, self.split)

            if not os.path.isdir(tf_split_dir):
                self.skipped_tfs.append({
                    "tf_name": tf_name,
                    "reason": f"split directory not found: {tf_split_dir}",
                })
                continue

            seq_path = os.path.join(tf_split_dir, "sequence.txt")
            label_path = os.path.join(tf_split_dir, "label.txt")
            meta_path = os.path.join(tf_split_dir, "meta.tsv")
            ppm_path = os.path.join(self.ppm_root, f"{tf_name}.txt")

            if not os.path.isfile(seq_path):
                raise FileNotFoundError(f"Missing sequence.txt: {seq_path}")

            if not os.path.isfile(label_path):
                raise FileNotFoundError(f"Missing label.txt: {label_path}")

            if not os.path.isfile(ppm_path):
                msg = f"Missing PPM for TF={tf_name}: {ppm_path}"
                if require_ppm:
                    raise FileNotFoundError(msg)
                else:
                    self.skipped_tfs.append({
                        "tf_name": tf_name,
                        "reason": msg,
                    })
                    continue

            ppm = load_ppm_file(ppm_path)
            self.tf_to_ppm[tf_name] = ppm

            sequences = load_lines(seq_path)
            label_lines = load_lines(label_path)
            meta_rows = load_meta_tsv(meta_path)

            validate_sequences_and_labels(
                sequences=sequences,
                labels=label_lines,
                expected_len=self.length,
                tf_name=tf_name,
                split=self.split,
            )

            if len(meta_rows) not in [0, len(sequences)]:
                raise ValueError(
                    f"[{tf_name}/{self.split}] meta.tsv row count mismatch: "
                    f"{len(meta_rows)} vs {len(sequences)}"
                )

            n_total = len(sequences)
            if max_samples_per_tf is None:
                n_keep = n_total
            else:
                n_keep = min(n_total, int(max_samples_per_tf))

            n_pos = 0
            n_neg = 0

            for i in range(n_keep):
                seq = sequences[i].strip()
                seq = seq.upper() if self.uppercase else seq

                nt_labels = parse_nt_label_line(
                    label_lines[i],
                    expected_len=self.length,
                )

                seq_label = float(nt_labels.sum() > 0)

                if seq_label > 0:
                    n_pos += 1
                else:
                    n_neg += 1

                meta = meta_rows[i] if len(meta_rows) > 0 else {}

                sample_id = (
                    meta.get("sample_id")
                    or meta.get("id")
                    or f"{tf_name}_{self.split}_L{self.length}_{i:07d}"
                )

                self.samples.append({
                    "sample_id": sample_id,
                    "tf_name": tf_name,
                    "sequence": seq,
                    "seq_label": seq_label,
                    "nt_labels": nt_labels,
                    "meta": meta,
                })

            self.tf_names.append(tf_name)

            if self.verbose:
                print(
                    f"[Stage2Dataset] loaded TF={tf_name}, split={self.split}, "
                    f"n={n_keep}, pos={n_pos}, neg={n_neg}, ppm_shape={ppm.shape}"
                )

        self.tf_names = sorted(set(self.tf_names))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples loaded. data_root={data_root}, ppm_root={ppm_root}, "
                f"split={split}, length={length}, tf_names={tf_names}"
            )

        if self.verbose:
            print("=" * 80)
            print("[Stage2Dataset] summary")
            print("data_root:", self.data_root)
            print("ppm_root:", self.ppm_root)
            print("split:", self.split)
            print("length:", self.length)
            print("num_tfs:", len(self.tf_names))
            print("num_samples:", len(self.samples))
            print("skipped_tfs:", len(self.skipped_tfs))
            if len(self.skipped_tfs) > 0:
                print("skipped examples:", self.skipped_tfs[:5])
            print("=" * 80)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        tf_name = item["tf_name"]

        return {
            "sample_id": item["sample_id"],
            "tf_name": tf_name,
            "sequence": item["sequence"],
            "seq_label": float(item["seq_label"]),
            "nt_labels": item["nt_labels"].copy(),
            "tf_ppm": self.tf_to_ppm[tf_name].copy(),
            "meta": dict(item["meta"]),
        }

class Stage2Collator:
    """
    Stage 2 collator.

    输入：
        dataset samples

    输出：
        input_ids
        attention_mask
        offset_mapping
        seq_labels
        token_labels
        token_valid_mask
        nt_labels
        tf_names
        tf_ppm
        tf_ppm_mask
        tf_ppm_lengths
        sample_ids
        sequences
    """

    def __init__(
        self,
        tokenizer,
        dna_length: int = 100,
        tokenizer_max_length: Optional[int] = None,
        debug: bool = False,
    ):
        self.tokenizer = tokenizer
        self.dna_length = int(dna_length)
        self.tokenizer_max_length = tokenizer_max_length
        self.debug = debug

    def pad_tf_ppm(
        self,
        ppm_list: List[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ppm_list:
            list of [4, M_i]

        return:
            tf_ppm: [B, 4, M_max]
            tf_ppm_mask: [B, M_max], True for valid motif positions
            tf_ppm_lengths: [B]
        """
        lengths = [int(ppm.shape[1]) for ppm in ppm_list]
        max_len = max(lengths)
        B = len(ppm_list)

        tf_ppm = np.zeros((B, 4, max_len), dtype=np.float32)
        tf_ppm_mask = np.zeros((B, max_len), dtype=np.bool_)

        for i, ppm in enumerate(ppm_list):
            if ppm.shape[0] != 4:
                raise ValueError(f"Expected PPM shape [4, M], got {ppm.shape}")

            m = ppm.shape[1]
            tf_ppm[i, :, :m] = ppm
            tf_ppm_mask[i, :m] = True

        return (
            torch.tensor(tf_ppm, dtype=torch.float32),
            torch.tensor(tf_ppm_mask, dtype=torch.bool),
            torch.tensor(lengths, dtype=torch.long),
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        sample_ids = [x["sample_id"] for x in features]
        tf_names = [x["tf_name"] for x in features]
        sequences = [x["sequence"] for x in features]

        seq_labels = np.asarray(
            [x["seq_label"] for x in features],
            dtype=np.float32,
        )

        nt_labels = np.stack(
            [x["nt_labels"] for x in features],
            axis=0,
        ).astype(np.float32)

        ppm_list = [x["tf_ppm"] for x in features]

        tokenize_kwargs = dict(
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        if self.tokenizer_max_length is not None:
            tokenize_kwargs["max_length"] = int(self.tokenizer_max_length)

        encoded = self.tokenizer(
            sequences,
            **tokenize_kwargs,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        offset_mapping = encoded["offset_mapping"]

        B, T = input_ids.shape

        token_labels = np.zeros((B, T), dtype=np.float32)
        token_valid_mask = np.zeros((B, T), dtype=np.bool_)

        offset_np = offset_mapping.cpu().numpy()

        for i in range(B):
            offsets_i = [
                (int(offset_np[i, j, 0]), int(offset_np[i, j, 1]))
                for j in range(T)
            ]

            tok_lab, tok_mask = nt_labels_to_token_labels(
                offset_mapping=offsets_i,
                nt_labels=nt_labels[i],
            )

            token_labels[i, :] = tok_lab
            token_valid_mask[i, :] = tok_mask

        tf_ppm, tf_ppm_mask, tf_ppm_lengths = self.pad_tf_ppm(ppm_list)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,

            "seq_labels": torch.tensor(seq_labels, dtype=torch.float32),
            "token_labels": torch.tensor(token_labels, dtype=torch.float32),
            "token_valid_mask": torch.tensor(token_valid_mask, dtype=torch.bool),
            "nt_labels": torch.tensor(nt_labels, dtype=torch.float32),

            "tf_names": tf_names,
            "tf_ppm": tf_ppm,
            "tf_ppm_mask": tf_ppm_mask,
            "tf_ppm_lengths": tf_ppm_lengths,

            "sample_ids": sample_ids,
            "sequences": sequences,
        }

        if "token_type_ids" in encoded:
            batch["token_type_ids"] = encoded["token_type_ids"]

        if self.debug:
            print("=" * 80)
            print("[Stage2Collator debug]")
            for k, v in batch.items():
                if hasattr(v, "shape"):
                    print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
                elif isinstance(v, list):
                    print(f"{k}: list len={len(v)}, example={v[:3]}")
                else:
                    print(f"{k}: {type(v)}")
            print("=" * 80)

        return batch