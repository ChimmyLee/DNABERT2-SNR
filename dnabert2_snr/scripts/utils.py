#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch

from transformers import AutoTokenizer, AutoConfig, AutoModel

def load_dnabert2_tokenizer(model_path: str):
    """
    Load DNABERT-2 tokenizer.

    Important:
        use_fast=True is required/recommended because Stage 1 depends on
        return_offsets_mapping=True for BPE-token to nucleotide alignment.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

    print("[DNABERT2 tokenizer]")
    print("  model_path:", model_path)
    print("  tokenizer class:", tokenizer.__class__)
    print("  is_fast:", getattr(tokenizer, "is_fast", None))
    print("  vocab size:", len(tokenizer))
    print("  pad_token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("  cls_token:", tokenizer.cls_token, tokenizer.cls_token_id)
    print("  sep_token:", tokenizer.sep_token, tokenizer.sep_token_id)
    print("  unk_token:", tokenizer.unk_token, tokenizer.unk_token_id)
    print("  mask_token:", tokenizer.mask_token, tokenizer.mask_token_id)

    if not getattr(tokenizer, "is_fast", False):
        print(
            "[WARN] tokenizer.is_fast is False. "
            "return_offsets_mapping may not work."
        )

    return tokenizer

def load_dnabert2_backbone(
    model_path: str,
    disable_flash: bool = True,
):
    """
    Official Stage 1 DNABERT-2 backbone loading function.

    Do NOT use AutoModel.from_pretrained directly, because in your environment
    it may trigger meta tensor issues.

    Loading logic:
        1. AutoTokenizer.from_pretrained
        2. AutoConfig.from_pretrained
        3. AutoModel.from_config
        4. disable flash attention
        5. manual torch.load(pytorch_model.bin)
        6. remove 'bert.' prefix
        7. load_state_dict(strict=False)

    Returns:
        tokenizer, config, backbone
    """

    tokenizer = load_dnabert2_tokenizer(model_path)

    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Sync special token ids from tokenizer to config
    config.pad_token_id = tokenizer.pad_token_id
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.unk_token_id = tokenizer.unk_token_id
    config.mask_token_id = tokenizer.mask_token_id

    backbone = AutoModel.from_config(
        config,
        trust_remote_code=True,
    )

    if disable_flash:
        bert_module = sys.modules[backbone.__class__.__module__]
        if hasattr(bert_module, "flash_attn_qkvpacked_func"):
            bert_module.flash_attn_qkvpacked_func = None
            print("[DNABERT2] flash_attn_qkvpacked_func disabled.")
        else:
            print("[DNABERT2] flash_attn_qkvpacked_func not found, skip disabling.")

    ckpt_path = os.path.join(model_path, "pytorch_model.bin")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"pytorch_model.bin not found: {ckpt_path}")

    state_dict = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("bert."):
            backbone_state_dict[k[len("bert."):]] = v

    missing, unexpected = backbone.load_state_dict(
        backbone_state_dict,
        strict=False,
    )

    print("[DNABERT2 backbone load]")
    print("  missing keys:", missing)
    print("  unexpected keys:", unexpected)

    # Your previous tests showed missing pooler keys are expected and harmless.
    allowed_missing = {
        "pooler.dense.weight",
        "pooler.dense.bias",
    }

    unexpected_missing = [k for k in missing if k not in allowed_missing]
    if len(unexpected_missing) > 0:
        print("[WARN] Unexpected missing keys:")
        for k in unexpected_missing:
            print("  ", k)

    if len(unexpected) > 0:
        print("[WARN] Unexpected keys:")
        for k in unexpected:
            print("  ", k)

    return tokenizer, config, backbone