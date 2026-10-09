#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
import argparse
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# -----------------------------
# basic utils
# -----------------------------

def stable_int_hash(s: str, seed: int = 2025) -> int:
    x = f"{seed}::{s}".encode("utf-8")
    return int(hashlib.md5(x).hexdigest(), 16) % (2**32)

def make_rng(key: str, seed: int = 2025) -> random.Random:
    return random.Random(stable_int_hash(key, seed))

def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()

def gc_content(seq: str) -> float:
    seq = seq.upper()
    valid = [x for x in seq if x in "ACGT"]
    if len(valid) == 0:
        return 0.0
    gc = sum(1 for x in valid if x in "GC")
    return gc / len(valid)

def has_n(seq: str) -> bool:
    return "N" in seq.upper()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# -----------------------------
# fasta reader using .fai
# -----------------------------

class FastaReader:
    """
    Minimal random-access FASTA reader based on .fai.
    Coordinates are 0-based half-open: fetch(chrom, start, end).
    """

    def __init__(self, fasta_path: str, fai_path: str):
        self.fasta_path = fasta_path
        self.fai_path = fai_path
        self.index = {}
        with open(fai_path, "r") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                chrom = parts[0]
                length = int(parts[1])
                offset = int(parts[2])
                line_bases = int(parts[3])
                line_width = int(parts[4])
                self.index[chrom] = {
                    "length": length,
                    "offset": offset,
                    "line_bases": line_bases,
                    "line_width": line_width,
                }
        self.fh = open(fasta_path, "rb")

    def chrom_len(self, chrom: str) -> int:
        return self.index[chrom]["length"]

    def has_chrom(self, chrom: str) -> bool:
        return chrom in self.index

    def fetch(self, chrom: str, start: int, end: int) -> str:
        if chrom not in self.index:
            raise KeyError(f"chrom not found in fai: {chrom}")

        info = self.index[chrom]
        chrom_len = info["length"]

        if start < 0 or end > chrom_len or start >= end:
            raise ValueError(
                f"Invalid fetch interval: {chrom}:{start}-{end}, chrom_len={chrom_len}"
            )

        offset = info["offset"]
        line_bases = info["line_bases"]
        line_width = info["line_width"]

        chunks = []
        pos = start

        while pos < end:
            line_idx = pos // line_bases
            col = pos % line_bases

            file_pos = offset + line_idx * line_width + col
            read_len = min(end - pos, line_bases - col)

            self.fh.seek(file_pos)
            chunk = self.fh.read(read_len)
            chunks.append(chunk)

            pos += read_len

        return b"".join(chunks).decode("ascii").upper()

    def close(self):
        self.fh.close()

# -----------------------------
# data structures
# -----------------------------

@dataclass
class Locus:
    tf_name: str
    chrom: str
    start: int
    end: int
    strand: str
    name: str
    locus_id: str
    center: float
    cluster_id: Optional[str] = None
    split: Optional[str] = None

@dataclass
class Window250:
    chrom: str
    start: int
    end: int

@dataclass
class NegativeMatch:
    chrom: str
    window250_start: int
    window250_end: int
    strand: str
    anchor_start: int
    anchor_end: int
    neg_locus_id: str
    gc250: float

# -----------------------------
# BED parsing and clustering
# -----------------------------

def read_bed_for_tf(bed_path: str, tf_name: str, min_len: int = 1, max_len: int = 100) -> List[Locus]:
    loci = []
    skipped_bad = 0
    skipped_len = 0

    with open(bed_path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.rstrip("\n").split()
            if len(parts) < 3:
                skipped_bad += 1
                continue

            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])

            name = parts[3] if len(parts) >= 4 else f"{tf_name}_{i}"
            strand = parts[5] if len(parts) >= 6 else "+"

            if strand not in ["+", "-"]:
                strand = "+"

            length = end - start
            if length < min_len:
                skipped_bad += 1
                continue

            # 因为最短窗口是 100bp，第一版保持所有 length 使用同一批 locus
            if length > max_len:
                skipped_len += 1
                continue

            locus_id = f"hg38:{chrom}:{start}:{end}:{strand}:{tf_name}"
            center = (start + end) / 2.0

            loci.append(
                Locus(
                    tf_name=tf_name,
                    chrom=chrom,
                    start=start,
                    end=end,
                    strand=strand,
                    name=name,
                    locus_id=locus_id,
                    center=center,
                )
            )

    print(f"[{tf_name}] loaded loci: {len(loci)}, skipped_bad={skipped_bad}, skipped_len>{max_len}={skipped_len}")
    return loci

def assign_clusters(loci: List[Locus], distance: int = 250) -> List[Locus]:
    """
    Same TF only.
    Rule: sorted by chrom and center.
    Adjacent loci with center distance < 250 are put into the same cluster.
    """
    loci_sorted = sorted(loci, key=lambda x: (x.chrom, x.center, x.start, x.end))

    current = []
    cluster_idx = 0

    def flush_cluster(cluster: List[Locus], idx: int):
        if not cluster:
            return
        tf_name = cluster[0].tf_name
        chrom = cluster[0].chrom
        c_start = min(x.start for x in cluster)
        c_end = max(x.end for x in cluster)
        cluster_id = f"{tf_name}:{chrom}:{c_start}:{c_end}:cluster{idx}"
        for x in cluster:
            x.cluster_id = cluster_id

    last_chrom = None
    last_center = None

    for loc in loci_sorted:
        if not current:
            current = [loc]
            last_chrom = loc.chrom
            last_center = loc.center
            continue

        same_chrom = loc.chrom == last_chrom
        close_enough = same_chrom and ((loc.center - last_center) < distance)

        if close_enough:
            current.append(loc)
        else:
            flush_cluster(current, cluster_idx)
            cluster_idx += 1
            current = [loc]

        last_chrom = loc.chrom
        last_center = loc.center

    flush_cluster(current, cluster_idx)
    return loci_sorted

def split_clusters(loci: List[Locus], seed: int = 2025) -> List[Locus]:
    cluster_ids = sorted(set(x.cluster_id for x in loci))
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)

    n = len(cluster_ids)

    if n >= 10:
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
    elif n >= 3:
        n_train = max(1, int(n * 0.8))
        n_val = 1
    else:
        n_train = n
        n_val = 0

    train_set = set(cluster_ids[:n_train])
    val_set = set(cluster_ids[n_train:n_train + n_val])
    test_set = set(cluster_ids[n_train + n_val:])

    for loc in loci:
        if loc.cluster_id in train_set:
            loc.split = "train"
        elif loc.cluster_id in val_set:
            loc.split = "val"
        elif loc.cluster_id in test_set:
            loc.split = "unseen_test"
        else:
            raise RuntimeError("cluster split assignment failed")

    print(
        f"[{loci[0].tf_name if loci else 'NA'}] clusters={n}, "
        f"train={len(train_set)}, val={len(val_set)}, unseen_test={len(test_set)}"
    )
    return loci

# -----------------------------
# interval helpers
# -----------------------------

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]

    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged

def build_tf_interval_index(loci: List[Locus]) -> Dict[str, List[Tuple[int, int]]]:
    d = {}
    for loc in loci:
        d.setdefault(loc.chrom, []).append((loc.start, loc.end))
    return {chrom: merge_intervals(v) for chrom, v in d.items()}

def interval_overlaps_any(start: int, end: int, intervals: List[Tuple[int, int]]) -> bool:
    """
    Simple linear scan. For typical TF BED size this is acceptable for first version.
    If later too slow, replace with bisect.
    """
    for s, e in intervals:
        if e <= start:
            continue
        if s >= end:
            break
        return True
    return False

def make_window250_for_locus(
    loc: Locus,
    fasta: FastaReader,
    seed: int = 2025,
    window_len: int = 250,
) -> Optional[Window250]:
    chrom_len = fasta.chrom_len(loc.chrom)
    tfbs_len = loc.end - loc.start

    if chrom_len < window_len:
        return None

    if tfbs_len > window_len:
        return None

    flank_total = window_len - tfbs_len
    rng = make_rng(f"pos250::{loc.locus_id}", seed)
    left_flank = rng.randint(0, flank_total)
    right_flank = flank_total - left_flank

    ws = loc.start - left_flank
    we = loc.end + right_flank

    # boundary correction
    if ws < 0:
        ws = 0
        we = window_len
    if we > chrom_len:
        we = chrom_len
        ws = chrom_len - window_len

    if ws < 0 or we > chrom_len or we - ws != window_len:
        return None

    if not (ws <= loc.start and loc.end <= we):
        return None

    return Window250(chrom=loc.chrom, start=ws, end=we)

def crop_from_250_containing_anchor(
    w250_start: int,
    w250_end: int,
    anchor_start: int,
    anchor_end: int,
    length: int,
    key: str,
    seed: int = 2025,
) -> Tuple[int, int]:
    valid_min = max(w250_start, anchor_end - length)
    valid_max = min(anchor_start, w250_end - length)

    if valid_min > valid_max:
        raise ValueError(
            f"No valid crop: w250={w250_start}-{w250_end}, "
            f"anchor={anchor_start}-{anchor_end}, length={length}, "
            f"valid={valid_min}-{valid_max}"
        )

    rng = make_rng(f"crop::{length}::{key}", seed)
    ws = rng.randint(valid_min, valid_max)
    return ws, ws + length

def build_label_for_window(
    chrom: str,
    window_start: int,
    window_end: int,
    intervals_by_chrom: Dict[str, List[Tuple[int, int]]],
) -> List[int]:
    L = window_end - window_start
    label = [0] * L

    for s, e in intervals_by_chrom.get(chrom, []):
        if e <= window_start:
            continue
        if s >= window_end:
            break

        os_ = max(s, window_start)
        oe_ = min(e, window_end)

        rel_s = os_ - window_start
        rel_e = oe_ - window_start

        for i in range(rel_s, rel_e):
            label[i] = 1

    return label

# -----------------------------
# negative sampling
# -----------------------------

def sample_negative_match(
    pos_loc: Locus,
    pos_w250: Window250,
    pos_seq250_raw: str,
    fasta: FastaReader,
    intervals_by_chrom: Dict[str, List[Tuple[int, int]]],
    seed: int = 2025,
    window_len: int = 250,
    strict_gc_diff: float = 0.02,
    relaxed_gc_diff: float = 0.05,
    max_tries: int = 20000,
) -> Optional[NegativeMatch]:

    chrom = pos_loc.chrom
    chrom_len = fasta.chrom_len(chrom)

    if chrom_len < window_len:
        return None

    pos_gc = gc_content(pos_seq250_raw)
    tfbs_len = pos_loc.end - pos_loc.start

    # preserve anchor relative location in the 250bp window
    pos_anchor_offset = pos_loc.start - pos_w250.start

    rng = make_rng(f"negative::{pos_loc.locus_id}", seed)

    intervals = intervals_by_chrom.get(chrom, [])

    for t in range(max_tries):
        gc_threshold = strict_gc_diff if t < max_tries // 2 else relaxed_gc_diff

        cand_start = rng.randint(0, chrom_len - window_len)
        cand_end = cand_start + window_len

        # must not overlap same-TF positive BED intervals
        if interval_overlaps_any(cand_start, cand_end, intervals):
            continue

        try:
            seq = fasta.fetch(chrom, cand_start, cand_end)
        except Exception:
            continue

        if has_n(seq):
            continue

        cand_gc = gc_content(seq)

        if abs(cand_gc - pos_gc) > gc_threshold:
            continue

        neg_anchor_start = cand_start + pos_anchor_offset
        neg_anchor_end = neg_anchor_start + tfbs_len

        # fake anchor must be inside negative 250 window
        if not (cand_start <= neg_anchor_start and neg_anchor_end <= cand_end):
            continue

        neg_strand = rng.choice(["+", "-"])
        neg_locus_id = (
            f"hg38:{chrom}:{neg_anchor_start}:{neg_anchor_end}:{neg_strand}:"
            f"{pos_loc.tf_name}:negative_for:{pos_loc.locus_id}"
        )

        return NegativeMatch(
            chrom=chrom,
            window250_start=cand_start,
            window250_end=cand_end,
            strand=neg_strand,
            anchor_start=neg_anchor_start,
            anchor_end=neg_anchor_end,
            neg_locus_id=neg_locus_id,
            gc250=cand_gc,
        )

    return None

# -----------------------------
# output
# -----------------------------

class OutputManager:
    def __init__(self, out_root: str, tf_name: str, lengths: List[int]):
        self.out_root = out_root
        self.tf_name = tf_name
        self.lengths = lengths
        self.handles = {}
        self.counts = {}

        for L in lengths:
            for split in ["train", "val", "unseen_test"]:
                d = os.path.join(out_root, str(L), tf_name, split)
                ensure_dir(d)

                seq_f = open(os.path.join(d, "sequence.txt"), "w")
                lab_f = open(os.path.join(d, "label.txt"), "w")
                meta_f = open(os.path.join(d, "meta.tsv"), "w")

                meta_header = [
                    "sample_id",
                    "tf_name",
                    "split",
                    "length",
                    "source",
                    "seq_label",
                    "chrom",
                    "window_start",
                    "window_end",
                    "strand",
                    "source_locus_id",
                    "matched_positive_locus_id",
                    "cluster_id",
                    "anchor_start",
                    "anchor_end",
                    "bed_start",
                    "bed_end",
                    "gc250",
                ]
                meta_f.write("\t".join(meta_header) + "\n")

                self.handles[(L, split)] = (seq_f, lab_f, meta_f)
                self.counts[(L, split, "positive")] = 0
                self.counts[(L, split, "negative")] = 0

    def write_record(
        self,
        length: int,
        split: str,
        sequence: str,
        label: List[int],
        meta: Dict[str, str],
    ):
        seq_f, lab_f, meta_f = self.handles[(length, split)]

        assert len(sequence) == length, (len(sequence), length)
        assert len(label) == length, (len(label), length)

        seq_f.write(sequence.upper() + "\n")
        lab_f.write("".join(str(x) for x in label) + "\n")

        fields = [
            "sample_id",
            "tf_name",
            "split",
            "length",
            "source",
            "seq_label",
            "chrom",
            "window_start",
            "window_end",
            "strand",
            "source_locus_id",
            "matched_positive_locus_id",
            "cluster_id",
            "anchor_start",
            "anchor_end",
            "bed_start",
            "bed_end",
            "gc250",
        ]
        meta_f.write("\t".join(str(meta.get(k, "")) for k in fields) + "\n")

        self.counts[(length, split, meta["source"])] += 1

    def close(self):
        for hs in self.handles.values():
            for h in hs:
                h.close()

    def print_counts(self):
        print(f"\n[{self.tf_name}] output counts")
        for L in self.lengths:
            for split in ["train", "val", "unseen_test"]:
                p = self.counts[(L, split, "positive")]
                n = self.counts[(L, split, "negative")]
                print(f"  length={L}, split={split}, positive={p}, negative={n}")

# -----------------------------
# per-TF generation
# -----------------------------

def process_tf(
    tf_name: str,
    bed_path: str,
    fasta: FastaReader,
    out_root: str,
    lengths: List[int],
    seed: int,
):
    print(f"\n========== Processing TF: {tf_name} ==========")

    loci = read_bed_for_tf(
        bed_path=bed_path,
        tf_name=tf_name,
        min_len=1,
        max_len=min(lengths),
    )

    if len(loci) == 0:
        print(f"[{tf_name}] no valid loci, skip.")
        return

    # filter chrom not in fasta
    loci = [x for x in loci if fasta.has_chrom(x.chrom)]

    if len(loci) == 0:
        print(f"[{tf_name}] no loci on chromosomes found in fasta, skip.")
        return

    loci = assign_clusters(loci, distance=250)
    loci = split_clusters(loci, seed=seed)

    intervals_by_chrom = build_tf_interval_index(loci)

    out = OutputManager(out_root=out_root, tf_name=tf_name, lengths=lengths)

    skipped_boundary = 0
    skipped_n_pos = 0
    skipped_neg_fail = 0
    skipped_crop = 0
    pair_idx = 0

    for loc in loci:
        w250 = make_window250_for_locus(
            loc=loc,
            fasta=fasta,
            seed=seed,
            window_len=250,
        )

        if w250 is None:
            skipped_boundary += 1
            continue

        try:
            pos_seq250_raw = fasta.fetch(loc.chrom, w250.start, w250.end)
        except Exception:
            skipped_boundary += 1
            continue

        if has_n(pos_seq250_raw):
            skipped_n_pos += 1
            continue

        neg = sample_negative_match(
            pos_loc=loc,
            pos_w250=w250,
            pos_seq250_raw=pos_seq250_raw,
            fasta=fasta,
            intervals_by_chrom=intervals_by_chrom,
            seed=seed,
            window_len=250,
        )

        if neg is None:
            skipped_neg_fail += 1
            continue

        pair_idx += 1

        # write positive and matched negative for each length
        for L in lengths:
            # ---------------- positive ----------------
            try:
                p_ws, p_we = crop_from_250_containing_anchor(
                    w250_start=w250.start,
                    w250_end=w250.end,
                    anchor_start=loc.start,
                    anchor_end=loc.end,
                    length=L,
                    key=f"positive::{loc.locus_id}",
                    seed=seed,
                )
            except Exception:
                skipped_crop += 1
                continue

            p_seq = fasta.fetch(loc.chrom, p_ws, p_we)
            p_label = build_label_for_window(
                chrom=loc.chrom,
                window_start=p_ws,
                window_end=p_we,
                intervals_by_chrom=intervals_by_chrom,
            )

            if loc.strand == "-":
                p_seq = reverse_complement(p_seq)
                p_label = p_label[::-1]

            p_sample_id = f"{tf_name}_{loc.split}_positive_{pair_idx:07d}_L{L}"

            p_meta = {
                "sample_id": p_sample_id,
                "tf_name": tf_name,
                "split": loc.split,
                "length": L,
                "source": "positive",
                "seq_label": 1,
                "chrom": loc.chrom,
                "window_start": p_ws,
                "window_end": p_we,
                "strand": loc.strand,
                "source_locus_id": loc.locus_id,
                "matched_positive_locus_id": "",
                "cluster_id": loc.cluster_id,
                "anchor_start": loc.start,
                "anchor_end": loc.end,
                "bed_start": loc.start,
                "bed_end": loc.end,
                "gc250": f"{gc_content(pos_seq250_raw):.6f}",
            }

            out.write_record(
                length=L,
                split=loc.split,
                sequence=p_seq,
                label=p_label,
                meta=p_meta,
            )

            # ---------------- negative ----------------
            try:
                n_ws, n_we = crop_from_250_containing_anchor(
                    w250_start=neg.window250_start,
                    w250_end=neg.window250_end,
                    anchor_start=neg.anchor_start,
                    anchor_end=neg.anchor_end,
                    length=L,
                    key=f"negative::{neg.neg_locus_id}",
                    seed=seed,
                )
            except Exception:
                skipped_crop += 1
                continue

            n_seq = fasta.fetch(neg.chrom, n_ws, n_we)
            n_label = [0] * L

            if neg.strand == "-":
                n_seq = reverse_complement(n_seq)
                n_label = n_label[::-1]

            n_sample_id = f"{tf_name}_{loc.split}_negative_{pair_idx:07d}_L{L}"

            n_meta = {
                "sample_id": n_sample_id,
                "tf_name": tf_name,
                "split": loc.split,
                "length": L,
                "source": "negative",
                "seq_label": 0,
                "chrom": neg.chrom,
                "window_start": n_ws,
                "window_end": n_we,
                "strand": neg.strand,
                "source_locus_id": neg.neg_locus_id,
                "matched_positive_locus_id": loc.locus_id,
                "cluster_id": loc.cluster_id,
                "anchor_start": neg.anchor_start,
                "anchor_end": neg.anchor_end,
                "bed_start": "",
                "bed_end": "",
                "gc250": f"{neg.gc250:.6f}",
            }

            out.write_record(
                length=L,
                split=loc.split,
                sequence=n_seq,
                label=n_label,
                meta=n_meta,
            )

    out.close()
    out.print_counts()

    print(
        f"[{tf_name}] skipped_boundary={skipped_boundary}, "
        f"skipped_n_pos={skipped_n_pos}, "
        f"skipped_neg_fail={skipped_neg_fail}, "
        f"skipped_crop={skipped_crop}"
    )

# -----------------------------
# main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bed_dir",
        type=str,
        default="../Integrated_TFBS_bed",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default="../hg38.fa",
    )
    parser.add_argument(
        "--fai",
        type=str,
        default="../hg38.fa.fai",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="/bed_result_split",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="100,150,200,250",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
    )
    parser.add_argument(
        "--tf",
        type=str,
        default=None,
        help="Run only one TF, e.g. --tf AR. If None, run all *.bed in bed_dir.",
    )

    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    lengths = sorted(lengths)

    print("bed_dir:", args.bed_dir)
    print("fasta:", args.fasta)
    print("fai:", args.fai)
    print("out_root:", args.out_root)
    print("lengths:", lengths)
    print("seed:", args.seed)

    fasta = FastaReader(args.fasta, args.fai)

    if args.tf is not None:
        bed_files = [os.path.join(args.bed_dir, f"{args.tf}.bed")]
    else:
        bed_files = sorted(glob.glob(os.path.join(args.bed_dir, "*.bed")))

    print("num bed files:", len(bed_files))

    for bed_path in bed_files:
        if not os.path.exists(bed_path):
            print(f"[WARN] bed not found: {bed_path}")
            continue

        tf_name = os.path.basename(bed_path)
        if tf_name.endswith(".bed"):
            tf_name = tf_name[:-4]

        process_tf(
            tf_name=tf_name,
            bed_path=bed_path,
            fasta=fasta,
            out_root=args.out_root,
            lengths=lengths,
            seed=args.seed,
        )

    fasta.close()

if __name__ == "__main__":
    main()
