#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_dnabert2_backbone

def masked_mean_pool(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Masked mean pooling.

    Args:
        x:
            Tensor, e.g. [B, T, H] or [B, M, D]
        mask:
            Bool or 0/1 tensor.
            For x=[B, T, H], mask should be [B, T].
            For x=[B, M, D], mask should be [B, M].
        dim:
            pooling dimension.

    Returns:
        pooled tensor.
    """
    if mask.dtype != torch.float32:
        mask = mask.float()

    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)

    x = x * mask
    denom = mask.sum(dim=dim).clamp_min(eps)

    return x.sum(dim=dim) / denom

def token_logits_to_nt_logits(
    token_logits: torch.Tensor,
    offset_mapping: torch.Tensor,
    attention_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    Expand token-level logits to nucleotide-level logits using offset_mapping.

    Args:
        token_logits:
            [B, T]
        offset_mapping:
            [B, T, 2], each token maps to nucleotide span [start, end)
        attention_mask:
            [B, T]
        seq_len:
            target nucleotide length, e.g. 100

    Returns:
        nt_logits:
            [B, seq_len]
    """
    device = token_logits.device
    dtype = token_logits.dtype

    B, T = token_logits.shape

    nt_logits_sum = torch.zeros((B, seq_len), device=device, dtype=dtype)
    nt_logits_cnt = torch.zeros((B, seq_len), device=device, dtype=dtype)

    for b in range(B):
        for t in range(T):
            if attention_mask[b, t].item() == 0:
                continue

            start = int(offset_mapping[b, t, 0].item())
            end = int(offset_mapping[b, t, 1].item())

            # Skip special tokens such as [CLS], [SEP], [PAD].
            if end <= start:
                continue

            start = max(0, min(start, seq_len))
            end = max(0, min(end, seq_len))

            if end <= start:
                continue

            nt_logits_sum[b, start:end] += token_logits[b, t]
            nt_logits_cnt[b, start:end] += 1.0

    nt_logits = nt_logits_sum / nt_logits_cnt.clamp_min(1.0)

    return nt_logits

def token_hidden_to_nt_hidden(
    token_hidden: torch.Tensor,
    offset_mapping: torch.Tensor,
    attention_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    Expand token-level hidden states to nucleotide-level hidden states
    using offset_mapping.

    Args:
        token_hidden:
            [B, T, H]
        offset_mapping:
            [B, T, 2], each token maps to nucleotide span [start, end)
        attention_mask:
            [B, T]
        seq_len:
            target nucleotide length

    Returns:
        nt_hidden:
            [B, seq_len, H]
    """
    device = token_hidden.device
    dtype = token_hidden.dtype

    B, T, H = token_hidden.shape

    nt_hidden_sum = torch.zeros((B, seq_len, H), device=device, dtype=dtype)
    nt_hidden_cnt = torch.zeros((B, seq_len, 1), device=device, dtype=dtype)

    for b in range(B):
        for t in range(T):
            if attention_mask[b, t].item() == 0:
                continue

            start = int(offset_mapping[b, t, 0].item())
            end = int(offset_mapping[b, t, 1].item())

            # Skip special tokens.
            if end <= start:
                continue

            start = max(0, min(start, seq_len))
            end = max(0, min(end, seq_len))

            if end <= start:
                continue

            nt_hidden_sum[b, start:end, :] += token_hidden[b, t].unsqueeze(0)
            nt_hidden_cnt[b, start:end, :] += 1.0

    nt_hidden = nt_hidden_sum / nt_hidden_cnt.clamp_min(1.0)

    return nt_hidden

class PPMEncoder(nn.Module):
    """
    Encode TF PPM into a global dense TF embedding.

    Input:
        tf_ppm:
            [B, 4, M]
        tf_ppm_mask:
            [B, M], True for valid motif positions

    Output:
        tf_embedding:
            [B, hidden_size]

    This is the original Stage 2 v1 PPM encoder.
    It is kept for:
        1. fusion_type="add"
        2. diagnostics
        3. backward compatibility
    """

    def __init__(
        self,
        hidden_size: int,
        ppm_channels: int = 4,
        conv_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=ppm_channels,
                out_channels=conv_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.proj = nn.Sequential(
            nn.Linear(conv_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tf_ppm: torch.Tensor,
        tf_ppm_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tf_ppm:
                [B, 4, M]
            tf_ppm_mask:
                [B, M]

        Returns:
            tf_embedding:
                [B, H]
        """
        if tf_ppm.ndim != 3:
            raise ValueError(
                f"Expected tf_ppm shape [B, 4, M], got {tf_ppm.shape}"
            )

        if tf_ppm.shape[1] != 4:
            raise ValueError(
                f"Expected tf_ppm channel dimension = 4, got {tf_ppm.shape}"
            )

        if tf_ppm_mask.ndim != 2:
            raise ValueError(
                f"Expected tf_ppm_mask shape [B, M], got {tf_ppm_mask.shape}"
            )

        if tf_ppm.shape[0] != tf_ppm_mask.shape[0]:
            raise ValueError(
                f"Batch size mismatch: tf_ppm={tf_ppm.shape}, "
                f"tf_ppm_mask={tf_ppm_mask.shape}"
            )

        if tf_ppm.shape[2] != tf_ppm_mask.shape[1]:
            raise ValueError(
                f"Motif length mismatch: tf_ppm={tf_ppm.shape}, "
                f"tf_ppm_mask={tf_ppm_mask.shape}"
            )

        h = self.conv(tf_ppm)      # [B, D, M]
        h = h.transpose(1, 2)      # [B, M, D]

        pooled = masked_mean_pool(
            x=h,
            mask=tf_ppm_mask,
            dim=1,
        )  # [B, D]

        tf_embedding = self.proj(pooled)  # [B, H]

        return tf_embedding

class PPMPositionEncoder(nn.Module):
    """
    Encode TF PPM into position-wise motif embeddings.

    Input:
        tf_ppm:
            [B, 4, M]
        tf_ppm_mask:
            [B, M], True for valid motif positions

    Output:
        ppm_hidden:
            [B, M, H]

    This is used by Stage 2 v2 cross-attention:
        DNA hidden states are Query.
        PPM position embeddings are Key / Value.
    """

    def __init__(
        self,
        hidden_size: int,
        ppm_channels: int = 4,
        conv_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
        max_ppm_len: int = 128,
        use_pos_embedding: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=ppm_channels,
                out_channels=conv_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.proj = nn.Sequential(
            nn.Linear(conv_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        self.max_ppm_len = int(max_ppm_len)
        self.use_pos_embedding = bool(use_pos_embedding)

        if self.use_pos_embedding:
            self.pos_embedding = nn.Embedding(self.max_ppm_len, hidden_size)
        else:
            self.pos_embedding = None

    def forward(
        self,
        tf_ppm: torch.Tensor,
        tf_ppm_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tf_ppm:
                [B, 4, M]
            tf_ppm_mask:
                [B, M]

        Returns:
            ppm_hidden:
                [B, M, H]
        """
        if tf_ppm.ndim != 3:
            raise ValueError(
                f"Expected tf_ppm shape [B, 4, M], got {tf_ppm.shape}"
            )

        if tf_ppm.shape[1] != 4:
            raise ValueError(
                f"Expected tf_ppm channel dimension = 4, got {tf_ppm.shape}"
            )

        if tf_ppm_mask.ndim != 2:
            raise ValueError(
                f"Expected tf_ppm_mask shape [B, M], got {tf_ppm_mask.shape}"
            )

        if tf_ppm.shape[0] != tf_ppm_mask.shape[0]:
            raise ValueError(
                f"Batch size mismatch: tf_ppm={tf_ppm.shape}, "
                f"tf_ppm_mask={tf_ppm_mask.shape}"
            )

        if tf_ppm.shape[2] != tf_ppm_mask.shape[1]:
            raise ValueError(
                f"Motif length mismatch: tf_ppm={tf_ppm.shape}, "
                f"tf_ppm_mask={tf_ppm_mask.shape}"
            )

        B, _, M = tf_ppm.shape

        if M > self.max_ppm_len:
            raise ValueError(
                f"PPM length M={M} exceeds max_ppm_len={self.max_ppm_len}. "
                f"Please increase ppm_max_len."
            )

        h = self.conv(tf_ppm)      # [B, D, M]
        h = h.transpose(1, 2)      # [B, M, D]
        h = self.proj(h)           # [B, M, H]

        if self.use_pos_embedding:
            pos_ids = torch.arange(M, device=tf_ppm.device).unsqueeze(0).expand(B, M)
            h = h + self.pos_embedding(pos_ids)

        # Zero-out padded motif positions for numerical cleanliness.
        h = h * tf_ppm_mask.to(dtype=h.dtype).unsqueeze(-1)

        return h

class DNAPPMCrossAttentionBlock(nn.Module):
    """
    DNA-to-PPM cross-attention block.

    Query:
        dna_hidden [B, T, H]

    Key / Value:
        ppm_hidden [B, M, H]

    Mask:
        tf_ppm_mask [B, M], True for valid motif positions.

    Output:
        updated DNA hidden [B, T, H]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_dim: Optional[int] = None,
        use_ffn: bool = True,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}"
            )

        if ffn_dim is None:
            ffn_dim = hidden_size * 4

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.use_ffn = bool(use_ffn)

        if self.use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_size),
                nn.Dropout(dropout),
            )
            self.ln2 = nn.LayerNorm(hidden_size)
        else:
            self.ffn = None
            self.ln2 = None

    def forward(
        self,
        dna_hidden: torch.Tensor,
        ppm_hidden: torch.Tensor,
        tf_ppm_mask: torch.Tensor,
        need_weights: bool = False,
    ):
        """
        Args:
            dna_hidden:
                [B, T, H]
            ppm_hidden:
                [B, M, H]
            tf_ppm_mask:
                [B, M], True for valid PPM positions
            need_weights:
                Whether to return attention weights.

        Returns:
            updated_hidden:
                [B, T, H]
            attn_weights:
                If need_weights=True:
                    [B, num_heads, T, M] in recent PyTorch.
                Else:
                    None
        """
        if dna_hidden.ndim != 3:
            raise ValueError(f"Expected dna_hidden [B, T, H], got {dna_hidden.shape}")

        if ppm_hidden.ndim != 3:
            raise ValueError(f"Expected ppm_hidden [B, M, H], got {ppm_hidden.shape}")

        if tf_ppm_mask.ndim != 2:
            raise ValueError(f"Expected tf_ppm_mask [B, M], got {tf_ppm_mask.shape}")

        if dna_hidden.shape[0] != ppm_hidden.shape[0]:
            raise ValueError(
                f"Batch size mismatch: dna_hidden={dna_hidden.shape}, "
                f"ppm_hidden={ppm_hidden.shape}"
            )

        if ppm_hidden.shape[:2] != tf_ppm_mask.shape:
            raise ValueError(
                f"PPM mask mismatch: ppm_hidden={ppm_hidden.shape}, "
                f"tf_ppm_mask={tf_ppm_mask.shape}"
            )

        # nn.MultiheadAttention key_padding_mask:
        # True means this key position should be ignored.
        key_padding_mask = ~tf_ppm_mask.bool()  # [B, M]

        if need_weights:
            attn_out, attn_weights = self.cross_attn(
                query=dna_hidden,
                key=ppm_hidden,
                value=ppm_hidden,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        else:
            attn_out, attn_weights = self.cross_attn(
                query=dna_hidden,
                key=ppm_hidden,
                value=ppm_hidden,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            attn_weights = None

        x = self.ln1(dna_hidden + self.dropout(attn_out))

        if self.use_ffn:
            ffn_out = self.ffn(x)
            x = self.ln2(x + ffn_out)

        return x, attn_weights

class DNAPPMCrossAttentionFusion(nn.Module):
    """
    Stack multiple DNA-to-PPM cross-attention blocks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_dim: Optional[int] = None,
        use_ffn: bool = True,
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.layers = nn.ModuleList(
            [
                DNAPPMCrossAttentionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_dim=ffn_dim,
                    use_ffn=use_ffn,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        dna_hidden: torch.Tensor,
        ppm_hidden: torch.Tensor,
        tf_ppm_mask: torch.Tensor,
        need_weights: bool = False,
    ):
        all_attn_weights: List[torch.Tensor] = []

        x = dna_hidden

        for layer in self.layers:
            x, attn_weights = layer(
                dna_hidden=x,
                ppm_hidden=ppm_hidden,
                tf_ppm_mask=tf_ppm_mask,
                need_weights=need_weights,
            )

            if need_weights:
                all_attn_weights.append(attn_weights)

        if need_weights:
            return x, all_attn_weights
        else:
            return x, None

class DNABERT2SNRStage2(nn.Module):
    """
    DNABERT2-SNR Stage 2 model.

    Stage 2 v1:
        fusion_type="add"

        hidden_states:
            [B, T, H]
        tf_embedding:
            [B, H]

        conditioned_hidden = LayerNorm(hidden_states + tf_embedding[:, None, :])

    Stage 2 v2:
        fusion_type="cross_attn"

        hidden_states:
            [B, T, H]
        ppm_hidden:
            [B, M, H]

        Query = DNA hidden states
        Key   = PPM position embeddings
        Value = PPM position embeddings

        conditioned_hidden = CrossAttention(DNA, PPM)

    Forward input ports are kept compatible with the original version.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        seq_len: int = 100,
        dna_length: Optional[int] = None,
        hidden_dropout_prob: float = 0.1,
        dropout: Optional[float] = None,
        loss_seq_weight: float = 1.0,
        loss_tok_weight: float = 1.0,
        loss_nt_weight: float = 1.0,
        seq_loss_weight: Optional[float] = None,
        token_loss_weight: Optional[float] = None,
        nt_loss_weight: Optional[float] = None,
        disable_flash: bool = True,
        freeze_backbone: bool = False,

        # PPM encoder options
        ppm_conv_dim: int = 128,
        ppm_kernel_size: int = 3,

        # Fusion options
        fusion_type: str = "cross_attn",
        ppm_max_len: int = 128,
        ppm_use_pos_embedding: bool = True,
        cross_attn_layers: int = 1,
        cross_attn_heads: int = 8,
        cross_attn_dropout: Optional[float] = None,
        cross_attn_ffn_dim: Optional[int] = None,
        cross_attn_use_ffn: bool = True,
        return_cross_attn_weights: bool = False,

        # token-level focal options
        token_focal_gamma: float = 0.0,
        token_focal_alpha: Optional[float] = None,

        # nucleotide-level focal + dice options
        nt_focal_gamma: float = 2.0,
        nt_focal_alpha: Optional[float] = 0.25,
        nt_focal_weight: float = 0.5,
        nt_dice_weight: float = 0.5,
        dice_smooth: float = 1.0,

        # independent nucleotide refinement head
        nt_head_type: str = "refine_conv",   # choices: "expand_logits", "refine_conv"
        nt_refine_dim: Optional[int] = None,
        nt_refine_kernel_size: int = 5,
        nt_refine_num_layers: int = 2,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Argument compatibility
        # ------------------------------------------------------------------
        if model_path is None and model_name_or_path is None:
            raise ValueError("Either model_path or model_name_or_path must be provided.")

        if model_path is None:
            model_path = model_name_or_path

        if dna_length is not None:
            seq_len = int(dna_length)

        if dropout is not None:
            hidden_dropout_prob = float(dropout)

        if seq_loss_weight is not None:
            loss_seq_weight = float(seq_loss_weight)

        if token_loss_weight is not None:
            loss_tok_weight = float(token_loss_weight)

        if nt_loss_weight is not None:
            loss_nt_weight = float(nt_loss_weight)

        fusion_type = str(fusion_type).lower().strip()
        if fusion_type not in ["add", "cross_attn"]:
            raise ValueError(
                f"Unsupported fusion_type={fusion_type}. "
                f"Expected 'add' or 'cross_attn'."
            )

        if cross_attn_dropout is None:
            cross_attn_dropout = hidden_dropout_prob

        self.model_path = model_path
        self.seq_len = int(seq_len)

        self.loss_seq_weight = float(loss_seq_weight)
        self.loss_tok_weight = float(loss_tok_weight)
        self.loss_nt_weight = float(loss_nt_weight)

        self.token_focal_gamma = token_focal_gamma
        self.token_focal_alpha = token_focal_alpha

        self.nt_focal_gamma = nt_focal_gamma
        self.nt_focal_alpha = nt_focal_alpha
        self.nt_focal_weight = nt_focal_weight
        self.nt_dice_weight = nt_dice_weight
        self.dice_smooth = dice_smooth

        # self.nt_head_type = str(nt_head_type).lower().strip()
        # if self.nt_head_type not in ["expand_logits", "refine_conv"]:
        #     raise ValueError(
        #         f"Unsupported nt_head_type={self.nt_head_type}. "
        #         f"Expected 'expand_logits' or 'refine_conv'."
        #     )

        # if nt_refine_dim is None:
        #     nt_refine_dim = max(self.hidden_size // 2, 64)

        # self.nt_refine_dim = int(nt_refine_dim)
        # self.nt_refine_kernel_size = int(nt_refine_kernel_size)
        # self.nt_refine_num_layers = int(nt_refine_num_layers)

        # if self.nt_refine_kernel_size <= 0 or self.nt_refine_kernel_size % 2 == 0:
        #     raise ValueError(
        #         f"nt_refine_kernel_size must be positive odd number, got {self.nt_refine_kernel_size}"
        #     )

        # if self.nt_refine_num_layers <= 0:
        #     raise ValueError(
        #         f"nt_refine_num_layers must be positive, got {self.nt_refine_num_layers}"
        #     )

        self.fusion_type = fusion_type
        self.return_cross_attn_weights = bool(return_cross_attn_weights)

        self.nt_head_type = str(nt_head_type).lower().strip()
        if self.nt_head_type not in ["expand_logits", "refine_conv"]:
            raise ValueError(
                f"Unsupported nt_head_type={self.nt_head_type}. "
                f"Expected 'expand_logits' or 'refine_conv'."
            )

# Do NOT use self.hidden_size here.
# self.hidden_size will be available only after loading the backbone.
        # ------------------------------------------------------------------
        # Load DNABERT-2 backbone exactly like Stage 1.
        # ------------------------------------------------------------------
        tokenizer, config, backbone = load_dnabert2_backbone(
            model_path=model_path,
            disable_flash=disable_flash,
        )

        self.tokenizer = tokenizer
        self.config = config
        self.backbone = backbone

        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("config.hidden_size is None, cannot build heads.")

        self.hidden_size = int(hidden_size)
        # ------------------------------------------------------------------
        # Independent nucleotide refinement head options
        # Must be initialized after self.hidden_size is available.
        # ------------------------------------------------------------------
        if nt_refine_dim is None:
            nt_refine_dim = max(self.hidden_size // 2, 64)

        self.nt_refine_dim = int(nt_refine_dim)
        self.nt_refine_kernel_size = int(nt_refine_kernel_size)
        self.nt_refine_num_layers = int(nt_refine_num_layers)

        if self.nt_refine_kernel_size <= 0 or self.nt_refine_kernel_size % 2 == 0:
            raise ValueError(
                f"nt_refine_kernel_size must be a positive odd number, "
                f"got {self.nt_refine_kernel_size}"
            )

        if self.nt_refine_num_layers <= 0:
            raise ValueError(
                f"nt_refine_num_layers must be positive, "
                f"got {self.nt_refine_num_layers}"
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(hidden_dropout_prob)

        # ------------------------------------------------------------------
        # PPM encoders
        # ------------------------------------------------------------------
        # Global PPM embedding, kept for diagnostics and fusion_type="add".
        self.ppm_encoder = PPMEncoder(
            hidden_size=self.hidden_size,
            ppm_channels=4,
            conv_dim=ppm_conv_dim,
            kernel_size=ppm_kernel_size,
            dropout=hidden_dropout_prob,
        )

        # Position-wise PPM embeddings, used by fusion_type="cross_attn".
        self.ppm_position_encoder = PPMPositionEncoder(
            hidden_size=self.hidden_size,
            ppm_channels=4,
            conv_dim=ppm_conv_dim,
            kernel_size=ppm_kernel_size,
            dropout=hidden_dropout_prob,
            max_ppm_len=ppm_max_len,
            use_pos_embedding=ppm_use_pos_embedding,
        )

        # ------------------------------------------------------------------
        # Fusion modules
        # ------------------------------------------------------------------
        self.fusion_ln = nn.LayerNorm(self.hidden_size)

        if self.fusion_type == "cross_attn":
            self.cross_attention_fusion = DNAPPMCrossAttentionFusion(
                hidden_size=self.hidden_size,
                num_layers=cross_attn_layers,
                num_heads=cross_attn_heads,
                dropout=cross_attn_dropout,
                ffn_dim=cross_attn_ffn_dim,
                use_ffn=cross_attn_use_ffn,
            )
        else:
            self.cross_attention_fusion = None

        # ------------------------------------------------------------------
        # Prediction heads
        # ------------------------------------------------------------------
        self.seq_classifier = nn.Linear(self.hidden_size, 1)
        self.token_classifier = nn.Linear(self.hidden_size, 1)
        # Independent nucleotide refinement head.
        # It operates on nucleotide-expanded hidden states instead of token logits.
        if self.nt_head_type == "refine_conv":
            nt_layers = []

            in_ch = self.hidden_size
            hidden_ch = self.nt_refine_dim
            k = self.nt_refine_kernel_size
            p = k // 2

            # first conv
            nt_layers.extend([
                nn.Conv1d(in_ch, hidden_ch, kernel_size=k, padding=p),
                nn.GELU(),
                nn.Dropout(hidden_dropout_prob),
            ])

            # middle conv blocks
            for _ in range(self.nt_refine_num_layers - 1):
                nt_layers.extend([
                    nn.Conv1d(hidden_ch, hidden_ch, kernel_size=k, padding=p),
                    nn.GELU(),
                    nn.Dropout(hidden_dropout_prob),
                ])

            # output conv
            nt_layers.append(nn.Conv1d(hidden_ch, 1, kernel_size=1))

            self.nt_refine_head = nn.Sequential(*nt_layers)
        else:
            self.nt_refine_head = None

    @property
    def dna_length(self) -> int:
        """
        Compatibility property for code that uses model.dna_length.
        """
        return self.seq_len

    def get_backbone_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return last hidden states from DNABERT-2 backbone.

        Expected:
            hidden_states [B, T, H]
        """
        backbone_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids

        outputs = self.backbone(**backbone_kwargs)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        return hidden_states

    def token_logits_to_nt_logits(
        self,
        token_logits: torch.Tensor,
        offset_mapping: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Expand token-level logits to nucleotide-level logits using offset_mapping.
        """
        if seq_len is None:
            seq_len = self.seq_len

        return token_logits_to_nt_logits(
            token_logits=token_logits,
            offset_mapping=offset_mapping,
            attention_mask=attention_mask,
            seq_len=seq_len,
        )

    def token_hidden_to_nt_hidden(
        self,
        token_hidden: torch.Tensor,
        offset_mapping: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Expand token-level hidden states to nucleotide-level hidden states
        using offset_mapping.
        """
        if seq_len is None:
            seq_len = self.seq_len

        return token_hidden_to_nt_hidden(
            token_hidden=token_hidden,
            offset_mapping=offset_mapping,
            attention_mask=attention_mask,
            seq_len=seq_len,
        )

    def build_nt_logits(
        self,
        token_hidden: torch.Tensor,
        token_logits: torch.Tensor,
        offset_mapping: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Build nucleotide-level logits using the configured nt head.

        expand_logits:
            token_logits -> nt_logits

        refine_conv:
            token_hidden -> nt_hidden -> conv refinement -> nt_logits
        """
        if self.nt_head_type == "expand_logits":
            nt_logits = self.token_logits_to_nt_logits(
                token_logits=token_logits,
                offset_mapping=offset_mapping,
                attention_mask=attention_mask,
                seq_len=seq_len,
            )
            return nt_logits

        elif self.nt_head_type == "refine_conv":
            nt_hidden = self.token_hidden_to_nt_hidden(
                token_hidden=token_hidden,
                offset_mapping=offset_mapping,
                attention_mask=attention_mask,
                seq_len=seq_len,
            )  # [B, L, H]

            nt_hidden = nt_hidden.transpose(1, 2)  # [B, H, L]
            nt_logits = self.nt_refine_head(nt_hidden).squeeze(1)  # [B, L]
            return nt_logits

        else:
            raise RuntimeError(f"Unsupported nt_head_type={self.nt_head_type}")

    @staticmethod
    def focal_bce_with_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Binary focal loss based on BCEWithLogitsLoss.

        Supports hard labels and soft labels in [0, 1].
        """
        targets = targets.to(dtype=logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        if gamma is None or gamma <= 0:
            loss = bce
        else:
            probs = torch.sigmoid(logits)
            p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
            focal_weight = torch.pow((1.0 - p_t).clamp_min(1e-8), gamma)

            if alpha is not None:
                alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
                focal_weight = alpha_t * focal_weight

            loss = focal_weight * bce

        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    @staticmethod
    def dice_loss_with_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Binary Dice loss from logits.

        For [B, L], Dice is computed per sample and then averaged.
        """
        targets = targets.to(dtype=logits.dtype)
        probs = torch.sigmoid(logits)

        if logits.ndim == 1:
            probs_flat = probs.view(1, -1)
            targets_flat = targets.view(1, -1)
        else:
            probs_flat = probs.view(probs.shape[0], -1)
            targets_flat = targets.view(targets.shape[0], -1)

        intersection = torch.sum(probs_flat * targets_flat, dim=1)
        pred_sum = torch.sum(probs_flat, dim=1)
        target_sum = torch.sum(targets_flat, dim=1)

        dice = (2.0 * intersection + smooth) / (
            pred_sum + target_sum + smooth
        ).clamp_min(1e-8)

        loss = 1.0 - dice

        if reduction == "mean":
            return loss.mean()
        elif reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def compute_loss(
        self,
        seq_logits: torch.Tensor,
        token_logits: torch.Tensor,
        nt_logits: Optional[torch.Tensor],
        seq_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        token_valid_mask: Optional[torch.Tensor] = None,
        nt_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Stage 2 losses.

        loss_seq:
            BCEWithLogitsLoss over sequence labels, shape [B].

        loss_tok:
            Masked soft BCEWithLogitsLoss over valid BPE tokens.
            If token_focal_gamma > 0, apply focal modulation.

        loss_nt:
            Focal loss + Dice loss over nucleotide labels, shape [B, L].
        """
        device = seq_logits.device
        dtype = seq_logits.dtype

        zero = torch.tensor(0.0, device=device, dtype=dtype)

        loss_seq = zero
        loss_tok = zero
        loss_nt = zero

        loss_tok_bce = zero
        loss_tok_focal = zero

        loss_nt_focal = zero
        loss_nt_dice = zero

        # -------------------------
        # sequence-level BCE
        # -------------------------
        if seq_labels is not None:
            seq_labels = seq_labels.to(device=device, dtype=seq_logits.dtype)

            loss_seq = F.binary_cross_entropy_with_logits(
                seq_logits,
                seq_labels,
                reduction="mean",
            )

        # -------------------------
        # token-level soft BCE + optional focal
        # -------------------------
        if token_labels is not None and token_valid_mask is not None:
            token_labels = token_labels.to(device=device, dtype=token_logits.dtype)
            token_valid_mask = token_valid_mask.to(device=device, dtype=torch.bool)

            if token_valid_mask.any():
                valid_token_logits = token_logits[token_valid_mask]
                valid_token_labels = token_labels[token_valid_mask]

                loss_tok_bce = F.binary_cross_entropy_with_logits(
                    valid_token_logits,
                    valid_token_labels,
                    reduction="mean",
                )

                loss_tok_focal = self.focal_bce_with_logits(
                    logits=valid_token_logits,
                    targets=valid_token_labels,
                    gamma=self.token_focal_gamma,
                    alpha=self.token_focal_alpha,
                    reduction="mean",
                )

                loss_tok = loss_tok_focal
            else:
                loss_tok_bce = zero
                loss_tok_focal = zero
                loss_tok = zero

        # -------------------------
        # nucleotide-level focal + dice
        # -------------------------
        if nt_labels is not None and nt_logits is not None:
            nt_labels = nt_labels.to(device=device, dtype=nt_logits.dtype)

            loss_nt_focal = self.focal_bce_with_logits(
                logits=nt_logits,
                targets=nt_labels,
                gamma=self.nt_focal_gamma,
                alpha=self.nt_focal_alpha,
                reduction="mean",
            )

            loss_nt_dice = self.dice_loss_with_logits(
                logits=nt_logits,
                targets=nt_labels,
                smooth=self.dice_smooth,
                reduction="mean",
            )

            loss_nt = (
                self.nt_focal_weight * loss_nt_focal
                + self.nt_dice_weight * loss_nt_dice
            )

        loss = (
            self.loss_seq_weight * loss_seq
            + self.loss_tok_weight * loss_tok
            + self.loss_nt_weight * loss_nt
        )

        return {
            "loss": loss,

            # Stage 1 / Stage 2 names
            "loss_seq": loss_seq,
            "loss_tok": loss_tok,
            "loss_nt": loss_nt,

            # Extra diagnostics
            "loss_tok_bce": loss_tok_bce,
            "loss_tok_focal": loss_tok_focal,
            "loss_nt_focal": loss_nt_focal,
            "loss_nt_dice": loss_nt_dice,

            # Aliases for earlier scripts
            "seq_loss": loss_seq,
            "token_loss": loss_tok,
            "nt_loss": loss_nt,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: torch.Tensor,
        tf_ppm: torch.Tensor,
        tf_ppm_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        seq_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
        token_valid_mask: Optional[torch.Tensor] = None,
        nt_labels: Optional[torch.Tensor] = None,
        tf_ppm_lengths: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids:
                [B, T]
            attention_mask:
                [B, T]
            offset_mapping:
                [B, T, 2]
            tf_ppm:
                [B, 4, M_max]
            tf_ppm_mask:
                [B, M_max]
            token_type_ids:
                optional [B, T]
            seq_labels:
                optional [B]
            token_labels:
                optional [B, T]
            token_valid_mask:
                optional [B, T]
            nt_labels:
                optional [B, L]
            tf_ppm_lengths:
                optional [B], currently unused.

        Returns:
            output dict with original fields plus:
                ppm_hidden
                cross_attn_weights, only if enabled
        """

        # Optional runtime override.
        # This keeps the forward port flexible without breaking existing calls.
        need_cross_attn_weights = kwargs.pop(
            "return_cross_attn_weights",
            self.return_cross_attn_weights,
        )

        # -------------------------
        # DNABERT-2 backbone
        # -------------------------
        hidden_states = self.get_backbone_hidden(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"Expected hidden_states to be [B, T, H], got {hidden_states.shape}"
            )

        # -------------------------
        # Global PPM embedding
        # -------------------------
        tf_embedding = self.ppm_encoder(
            tf_ppm=tf_ppm,
            tf_ppm_mask=tf_ppm_mask,
        )  # [B, H]

        if tf_embedding.shape[0] != hidden_states.shape[0]:
            raise RuntimeError(
                f"Batch size mismatch: hidden_states={hidden_states.shape}, "
                f"tf_embedding={tf_embedding.shape}"
            )

        if tf_embedding.shape[1] != hidden_states.shape[2]:
            raise RuntimeError(
                f"Hidden size mismatch: hidden_states={hidden_states.shape}, "
                f"tf_embedding={tf_embedding.shape}"
            )

        # -------------------------
        # Position-wise PPM embedding
        # -------------------------
        ppm_hidden = self.ppm_position_encoder(
            tf_ppm=tf_ppm,
            tf_ppm_mask=tf_ppm_mask,
        )  # [B, M, H]

        # -------------------------
        # TF-conditioned fusion
        # -------------------------
        cross_attn_weights = None

        if self.fusion_type == "add":
            # Stage 2 v1 behavior.
            conditioned_hidden = hidden_states + tf_embedding.unsqueeze(1)
            conditioned_hidden = self.fusion_ln(conditioned_hidden)

        elif self.fusion_type == "cross_attn":
            # Stage 2 v2 behavior.
            conditioned_hidden, cross_attn_weights = self.cross_attention_fusion(
                dna_hidden=hidden_states,
                ppm_hidden=ppm_hidden,
                tf_ppm_mask=tf_ppm_mask,
                need_weights=need_cross_attn_weights,
            )

        else:
            raise RuntimeError(f"Unsupported fusion_type={self.fusion_type}")

        # -------------------------
        # sequence-level prediction
        # -------------------------
        cls_hidden = conditioned_hidden[:, 0, :]  # [B, H]
        cls_hidden = self.dropout(cls_hidden)

        seq_logits = self.seq_classifier(cls_hidden).squeeze(-1)  # [B]

        # -------------------------
        # token-level prediction
        # -------------------------
        token_hidden = self.dropout(conditioned_hidden)
        token_logits = self.token_classifier(token_hidden).squeeze(-1)  # [B, T]

        # -------------------------
        # nucleotide-level prediction
        # -------------------------
        # Runtime DNA length support.
        # This is required for multi-length training, e.g. 100/150/200.
        runtime_seq_len = kwargs.pop("dna_length", None)

        if runtime_seq_len is None:
            if nt_labels is not None:
                runtime_seq_len = int(nt_labels.shape[1])
            else:
                runtime_seq_len = self.seq_len
        else:
            runtime_seq_len = int(runtime_seq_len)

        nt_logits = self.build_nt_logits(
            token_hidden=token_hidden,
            token_logits=token_logits,
            offset_mapping=offset_mapping,
            attention_mask=attention_mask,
            seq_len=runtime_seq_len,
        )  # [B, L]

        # -------------------------
        # losses
        # -------------------------
        loss_dict = self.compute_loss(
            seq_logits=seq_logits,
            token_logits=token_logits,
            nt_logits=nt_logits,
            seq_labels=seq_labels,
            token_labels=token_labels,
            token_valid_mask=token_valid_mask,
            nt_labels=nt_labels,
        )

        output = {
            "seq_logits": seq_logits,
            "token_logits": token_logits,
            "nt_logits": nt_logits,

            # Original diagnostics
            "tf_embedding": tf_embedding,
            "conditioned_hidden": conditioned_hidden,

            # New diagnostics for Stage 2 v2
            "ppm_hidden": ppm_hidden,
            "nt_head_type": self.nt_head_type,
        }

        if cross_attn_weights is not None:
            output["cross_attn_weights"] = cross_attn_weights

        output.update(loss_dict)

        return output

# Backward-compatible alias.
# If your test/train script imports Stage2TFBSModel, it will still work.
Stage2TFBSModel = DNABERT2SNRStage2