"""
arca/core/gnn_policy.py  (v3.5 — AMP-safe Dynamic Action Masking)
==================================================================
Fixes vs v3.4:
  PRIMARY FIX: _apply_mask now computes the fill value from logits.dtype
  instead of using the hardcoded constant _NEG_INF = -1e9.

  Why -1e9 crashes: torch.amp.autocast converts forward-pass tensors to
  float16 inside the autocast context.  float16 has a max magnitude of
  ~65504, so masked_fill(-1e9) raises:
      RuntimeError: value cannot be converted to type at::Half without overflow

  Fix: compute fill_value = torch.finfo(logits.dtype).min / 2 at runtime.
       float16 → -32752   (representable, causes softmax ≈ 0 on masked slots)
       float32 → ~-1.7e38 (very negative, same behaviour as before)
       bfloat16 → also safe

  The dynamic size-mismatch handling from v3.4 is preserved unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from torch_geometric.nn import (
        GCNConv, GATv2Conv,
        global_mean_pool, global_max_pool,
    )
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None
    Batch = None


class GNNEncoder(nn.Module):
    """
    3-layer GCN (or GATv2) encoder with dual graph-level pooling.

    Input:  PyG Data  (x: [N, feat], edge_index: [2, E])
    Output: Tensor    [B, hidden_dim * 2]
    """

    def __init__(
        self,
        feature_dim: int  = 9,
        hidden_dim:  int  = 128,
        use_gat:     bool = False,
    ):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError(
                "torch-geometric not installed. Run: pip install torch-geometric"
            )

        if use_gat:
            heads        = 4
            opH          = hidden_dim // heads
            self.conv1   = GATv2Conv(feature_dim, opH, heads=heads, concat=True)
            self.conv2   = GATv2Conv(hidden_dim,  opH, heads=heads, concat=True)
            self.conv3   = GATv2Conv(hidden_dim,  opH, heads=heads, concat=True)
        else:
            self.conv1 = GCNConv(feature_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim,  hidden_dim)
            self.conv3 = GCNConv(hidden_dim,  hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, data: "Data") -> torch.Tensor:
        x          = data.x
        edge_index = data.edge_index
        batch      = (
            data.batch
            if (hasattr(data, "batch") and data.batch is not None)
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        x = F.dropout(F.relu(self.ln1(self.conv1(x, edge_index))),
                       p=0.1, training=self.training)
        x = F.relu(self.ln2(self.conv2(x, edge_index)))
        x = F.relu(self.ln3(self.conv3(x, edge_index)))

        return torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)],
            dim=-1,
        )   # [B, hidden_dim * 2]


class GNNPolicy(nn.Module):
    """
    Masked actor-critic policy backed by a Graph Neural Network.

    All public methods accept an optional `action_mask` argument:
        action_mask : BoolTensor [batch, n_actions]  (True = valid)

    v3.5 fixes:
      - _apply_mask: fill value is computed from logits.dtype so it is
        always representable in float16 / bfloat16 (AMP-safe).
      - _apply_mask: size-mismatch handling from v3.4 preserved.
    """

    def __init__(
        self,
        feature_dim: int  = 9,
        hidden_dim:  int  = 128,
        num_actions: int  = 160,
        use_gat:     bool = False,
    ):
        super().__init__()
        self.encoder     = GNNEncoder(feature_dim, hidden_dim, use_gat)
        self.num_actions = num_actions
        embed_dim        = hidden_dim * 2

        self.actor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_mask(
        self,
        logits:      torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Apply an action mask to logits, handling two independent problems:

        Problem 1 — Size mismatch (curriculum learning, v3.4 fix):
          Each curriculum tier has a different number of hosts, so the
          action space changes (80 / 160 / 240 / 360 / 500 actions).
          When a model trained on tier N is evaluated on tier N+1 env,
          logits.shape[1] != mask.shape[-1].
          Resolution:
            n_mask > n_logits → truncate mask (extra actions unreachable)
            n_mask < n_logits → pad mask with False (extra logit slots masked)

        Problem 2 — AMP / float16 overflow (v3.5 fix, PRIMARY BUG):
          torch.amp.autocast converts logits to float16 inside the training
          loop.  float16 max magnitude is ~65504.  Using masked_fill(-1e9)
          raises "value cannot be converted to type at::Half without overflow".
          Resolution:
            Compute fill_value = torch.finfo(logits.dtype).min / 2 at runtime.
            float16  → fill ≈ -32752   (softmax ≈ 0, always representable)
            float32  → fill ≈ -1.7e38  (same behaviour as the old -1e9)
            bfloat16 → fill ≈ -1.7e38  (also safe)

        Args:
            logits:      [B, n_logits]   raw actor outputs (may be fp16 under AMP)
            action_mask: [B, n_mask] or [n_mask]  bool  (True = valid)
                         May be None → no masking applied.

        Returns:
            logits with invalid slots filled with a very-negative value.
        """
        if action_mask is None:
            return logits

        # Ensure mask has a batch dimension matching logits
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0).expand(logits.shape[0], -1)

        n_logits = logits.shape[-1]
        n_mask   = action_mask.shape[-1]

        # ── Problem 1: resize mask to match logits (curriculum fix) ───────────
        if n_logits != n_mask:
            if n_mask > n_logits:
                # Mask is LARGER → truncate (actions beyond policy head unreachable)
                action_mask = action_mask[..., :n_logits]
            else:
                # Mask is SMALLER → pad with False (extra logit slots masked out)
                pad = torch.zeros(
                    (*action_mask.shape[:-1], n_logits - n_mask),
                    dtype=torch.bool,
                    device=action_mask.device,
                )
                action_mask = torch.cat([action_mask, pad], dim=-1)

        # ── Problem 2: dtype-safe fill value (AMP / fp16 fix) ─────────────────
        # torch.finfo gives the representable minimum for any float dtype.
        # Dividing by 2 ensures we stay away from the exact minimum (which
        # can cause NaN in some ops) while still being extremely negative.
        fill_value = torch.finfo(logits.dtype).min / 2

        return logits.masked_fill(~action_mask, fill_value)

    def _masked_dist(
        self,
        data,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[Categorical, torch.Tensor]:
        """Return (masked Categorical distribution, value estimate)."""
        emb    = self.encoder(data)
        logits = self._apply_mask(self.actor(emb), action_mask)
        value  = self.critic(emb).squeeze(-1)
        return Categorical(logits=logits), value

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        data,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb    = self.encoder(data)
        logits = self._apply_mask(self.actor(emb), action_mask)
        value  = self.critic(emb).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        data,
        action:      torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self._masked_dist(data, action_mask)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(
        self,
        data,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, value = self._masked_dist(data, action_mask)
        return value

    def get_action(
        self,
        data,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample or argmax over the masked action distribution.

        deterministic=True  → argmax over valid actions (masked slots have
                               fill_value ≈ -32752 for fp16 or -1.7e38 for fp32,
                               so they are never selected)
        deterministic=False → sample from masked Categorical

        Both the size-mismatch (v3.4) and AMP overflow (v3.5) fixes apply
        transparently through _apply_mask, so this method is safe to call
        across all curriculum tiers under mixed-precision training.
        """
        dist, _ = self._masked_dist(data, action_mask)
        if deterministic:
            return dist.logits.argmax(dim=-1)
        return dist.sample()