"""OnsetAwareSetTransformer: set-attention ranker for RCA component scoring."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OnsetAwareSetTransformer(nn.Module):
    """Set Transformer that ranks candidates using onset positional encoding.

    Input:  (B, N_max, 33)  feature vectors for N candidates
            (B, N_max)       onset ranks (0=earliest)
    Output: (B, N_max)      logits (higher = more likely root cause)

    The onset order is encoded as a learned positional embedding that is
    added to the candidate features before the self-attention layers.
    """

    def __init__(
        self,
        n_features: int = 33,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        max_candidates: int = 100,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Onset rank → positional embedding
        self.onset_pos = nn.Embedding(max_candidates + 1, d_model, padding_idx=0)

        # A learned scalar that modulates how strongly onset influences the
        # attention via a separate cross-candidate onset gate.
        self.onset_gate = nn.Parameter(torch.tensor(0.5))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)

        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        onset_ranks: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (B, N, n_features) per-candidate feature vectors.
            onset_ranks: (B, N) onset rank (0=earliest).
            padding_mask: (B, N) True where padded.

        Returns:
            scores: (B, N) raw logits.
        """
        B, N, _ = features.shape

        h = self.feature_encoder(features)

        onset_ranks_clamped = onset_ranks.clamp(0, 99)
        h = h + self.onset_pos(onset_ranks_clamped)

        # Build cross-candidate onset features: a learned "onset salience" that
        # gets added before the transformer so the model sees which candidates
        # are early vs late.
        onset_salience = torch.sigmoid(self.onset_gate) * (1.0 / (onset_ranks.float() + 1.0))
        h = h + onset_salience.unsqueeze(-1)

        h = self.transformer(
            h,
            src_key_padding_mask=padding_mask,
        )

        scores = self.scorer(h).squeeze(-1)
        return scores

    def predict_proba(
        self,
        features: torch.Tensor,
        onset_ranks: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return per-candidate root-cause probabilities (sigmoid-normalized)."""
        logits = self.forward(features, onset_ranks, padding_mask)
        probs = torch.sigmoid(logits)
        if padding_mask is not None:
            probs = probs.masked_fill(padding_mask, 0.0)
        return probs

    @classmethod
    def from_pretrained_encoder(
        cls,
        encoder_state: dict[str, torch.Tensor],
        **kwargs,
    ) -> "OnsetAwareSetTransformer":
        model = cls(**kwargs)
        missing, unexpected = model.feature_encoder.load_state_dict(encoder_state, strict=False)
        return model
