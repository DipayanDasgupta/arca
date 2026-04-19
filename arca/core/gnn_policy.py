"""
arca/core/gnn_policy.py  (v3.2 — Action Masking)
==================================================
New in v3.2:
  - All sampling paths (train + eval) accept an optional `action_mask`
    boolean tensor of shape [batch, n_actions].
  - Masked logits set to -1e9 before Categorical → invalid actions
    never selected, log_prob never computed for them.
  - get_action() no longer uses top-k hack; pure argmax is correct
    once masking prevents invalid actions.
  - GNNEncoder unchanged (LayerNorm + dual pooling from v3.1).
  - Weight init unchanged (actor gain=0.01 for high initial entropy).
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

_NEG_INF = -1e9   # logit value for masked (invalid) actions


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

        # LayerNorm: stable for any batch size (critical for RL)
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
    When provided, invalid actions receive logit = -1e9 before any
    sampling or log_prob computation.
    """

    def __init__(
        self,
        feature_dim: int  = 9,
        hidden_dim:  int  = 128,
        num_actions: int  = 160,
        use_gat:     bool = False,
    ):
        super().__init__()
        self.encoder    = GNNEncoder(feature_dim, hidden_dim, use_gat)
        self.num_actions = num_actions
        embed_dim       = hidden_dim * 2

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
        # Small actor output → high initial entropy → more exploration
        nn.init.orthogonal_(self.actor[-1].weight,  gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_mask(
        self,
        logits: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Zero-out invalid action logits.
        logits      : [B, n_actions]
        action_mask : [B, n_actions]  bool  (True = valid)  or None
        """
        if action_mask is None:
            return logits
        # Broadcast scalar batch if needed
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0).expand_as(logits)
        return logits.masked_fill(~action_mask, _NEG_INF)

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
        deterministic=True  → argmax over valid actions (mask applied first)
        deterministic=False → sample from masked Categorical
        """
        dist, _ = self._masked_dist(data, action_mask)
        if deterministic:
            # argmax is safe now: invalid logits are -1e9
            return dist.logits.argmax(dim=-1)
        return dist.sample()