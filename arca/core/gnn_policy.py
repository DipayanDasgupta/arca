"""
arca/core/gnn_policy.py  (v3.1 — FIXED)
=========================================
Key fixes vs v3.0:
  1. BatchNorm1d → nn.LayerNorm throughout GNNEncoder.
     BatchNorm is fundamentally unstable when:
       - Called with a single graph (batch_size=1) during get_value()
       - Called during inference with .eval() on a single node
     LayerNorm normalises per-sample (not per-batch) — stable for any batch.

  2. Orthogonal weight init:
       - Hidden layers: gain=1.0  (standard for Tanh networks)
       - Actor output : gain=0.01 (small logits → nearly uniform initial policy
                                   → high entropy → better exploration)
       - Critic output: gain=1.0

Architecture:
  - 3-layer GCN (or GATv2) encoder
  - Dual pooling: mean-pool ⊕ max-pool → 2× hidden_dim embedding
  - Actor head: embedding → action logits
  - Critic head: embedding → scalar value
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None
    Batch = None


class GNNEncoder(nn.Module):
    """
    3-layer message-passing encoder with dual graph-level pooling.

    Input:  PyG Data  (x: [N, feature_dim], edge_index: [2, E])
    Output: Tensor    shape [batch_size, hidden_dim * 2]
    """

    def __init__(
        self,
        feature_dim: int = 9,
        hidden_dim:  int = 128,
        use_gat:     bool = False,
    ):
        super().__init__()
        if not PYG_AVAILABLE:
            raise ImportError(
                "torch-geometric not installed. "
                "Run: pip install torch-geometric"
            )

        if use_gat:
            heads        = 4
            out_per_head = hidden_dim // heads
            self.conv1   = GATv2Conv(feature_dim, out_per_head, heads=heads, concat=True)
            self.conv2   = GATv2Conv(hidden_dim,  out_per_head, heads=heads, concat=True)
            self.conv3   = GATv2Conv(hidden_dim,  out_per_head, heads=heads, concat=True)
        else:
            self.conv1 = GCNConv(feature_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim,  hidden_dim)
            self.conv3 = GCNConv(hidden_dim,  hidden_dim)

        # LayerNorm: stable for any batch size, including batch_size=1
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

        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.ln3(x)
        x = F.relu(x)

        # Dual pooling: mean ⊕ max → richer graph-level embedding
        x_mean = global_mean_pool(x, batch)   # [B, hidden_dim]
        x_max  = global_max_pool(x, batch)    # [B, hidden_dim]
        return torch.cat([x_mean, x_max], dim=-1)  # [B, hidden_dim * 2]


class GNNPolicy(nn.Module):
    """
    Actor-critic policy backed by a Graph Neural Network.

    Input  : PyG Data / Batch  (per-episode or batched)
    Outputs: action logits + scalar value
    """

    def __init__(
        self,
        feature_dim: int  = 9,
        hidden_dim:  int  = 128,
        num_actions: int  = 160,
        use_gat:     bool = False,
    ):
        super().__init__()
        self.encoder  = GNNEncoder(feature_dim, hidden_dim, use_gat)
        embed_dim     = hidden_dim * 2   # dual pooling

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
        # Hidden layers: gain=1.0 (proper for Tanh)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor output: tiny gain → nearly uniform initial policy → high entropy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Critic output: standard gain
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        emb    = self.encoder(data)
        logits = self.actor(emb)
        value  = self.critic(emb).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        data,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(data)
        dist          = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, data) -> torch.Tensor:
        return self.forward(data)[1]
    
    def get_action(self, data, deterministic: bool = False) -> torch.Tensor:
        """Get action from policy. Used during evaluation and deterministic rollouts."""
        logits, _ = self.forward(data)
        
        if deterministic:
            # Top-5 sampling prevents the agent from getting stuck on the same invalid action
            # (this was causing the constant -75 reward in evaluation)
            k = min(5, logits.size(-1))
            top_vals, top_idx = logits.topk(k, dim=-1)
            chosen = Categorical(logits=top_vals).sample()
            return top_idx.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        
        # Stochastic sampling (used during training rollouts)
        return Categorical(logits=logits).sample()