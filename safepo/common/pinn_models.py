# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Physics-Informed Neural Network (PINN) components for multi-agent systems."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron with SiLU activation."""
    
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            if i < len(self.hidden_channels) - 2:
                layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLP2(nn.Module):
    """Multi-Layer Perceptron with SiLU activation after each layer."""
    
    def __init__(self, in_channels, hidden_channels):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        layers = [nn.Linear(self.in_channels, self.hidden_channels[0]), nn.SiLU()]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(nn.Linear(self.hidden_channels[i], self.hidden_channels[i + 1]))
            layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Attention_LEMURS(nn.Module):
    """Attention mechanism for LEMURS (Learning from Demonstrations with Uncertainty-aware Residual Shaping)."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()

        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        # Initialized to avoid unstable training
        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(
            input_dim,
            [2 * hidden_dim]
        ).to(device)

        self.mlp_hidden_4 = MLP(
            2 * hidden_dim,
            [hidden_dim]
        ).to(device)

        self.mlp_out = MLP(
            hidden_dim,
            [output_dim]
        ).to(device)

    def forward(self, x):
        self.na = x.shape[1]
        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bk_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bv_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bk_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bv_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        return self.mlp_out(x.mean(dim=1)).reshape(-1, self.na, self.output_dim)


class Att_R(nn.Module):
    """Attention-based R matrix computation for port-Hamiltonian systems."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, na, scenario_name, device):
        super().__init__()
        self.device = device
        self.scenario_name = scenario_name
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)
        
        # Cache eye matrix for kron replacement
        self.register_buffer('_eye2', torch.eye(2, device=device))

    def forward(self, x, laplacian, scenario_name):
        self.na = x.shape[1]

        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(x.shape[0], self.na, -1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bk_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2)) + self.Bv_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)

        batch = int(x.shape[0] / x.shape[2])

        j12 = x.sum(1).sum(1).reshape(batch, self.na)
        j21 = -j12
        
        # Optimized matrix construction using diag_embed
        J12 = torch.diag_embed(j12)
        J21 = torch.diag_embed(j21)
        zeros = torch.zeros_like(J12)
        
        J = torch.cat((torch.cat((zeros, J21), dim=1), torch.cat((J12, zeros), dim=1)), dim=2)

        # Optimized kron replacement: J ⊗ I_2 using repeat_interleave
        # This is much faster than torch.kron for this specific pattern
        J_expanded = J.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        eye_pattern = self._eye2.unsqueeze(0).expand(J.shape[0], -1, -1)
        eye_tiled = eye_pattern.repeat(1, J.shape[1], J.shape[2])
        return J_expanded * eye_tiled


class Att_J(nn.Module):
    """Attention-based J matrix computation for port-Hamiltonian systems."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, na, scenario_name, device):
        super().__init__()
        self.device = device
        self.scenario_name = scenario_name
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)
        
        # Cache eye matrix for kron replacement
        self.register_buffer('_eye2', torch.eye(2, device=device))

    def forward(self, x, laplacian, scenario_name):
        batch_size = x.shape[0]
        self.na = x.shape[1]

        x = self.mlp_in(x.reshape(-1, self.input_dim)).reshape(batch_size, self.na, -1)
        x_t = x.transpose(1, 2)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(0).expand(batch_size, -1, -1), x_t) + self.Bq_4.unsqueeze(0))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(0).expand(batch_size, -1, -1), x_t) + self.Bk_4.unsqueeze(0)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(0).expand(batch_size, -1, -1), x_t) + self.Bv_4.unsqueeze(0))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).reshape(batch_size, self.na, -1)
        x_t = x.transpose(1, 2)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(0).expand(batch_size, -1, -1), x_t) + self.Bq_7.unsqueeze(0))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(0).expand(batch_size, -1, -1), x_t) + self.Bk_7.unsqueeze(0)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(0).expand(batch_size, -1, -1), x_t) + self.Bv_7.unsqueeze(0))

        x = self.activation_swish(
            torch.bmm(self.activation_soft(torch.bmm(Q, K)), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).reshape(-1, self.na, self.output_dim).transpose(1, 2)

        batch = x.shape[0] // x.shape[2]

        j12 = x.sum(1).sum(1).reshape(batch, self.na)
        j21 = -j12
        
        # Optimized matrix construction using diag_embed
        J12 = torch.diag_embed(j12)
        J21 = torch.diag_embed(j21)
        zeros = torch.zeros_like(J12)
        
        J = torch.cat((torch.cat((zeros, J21), dim=1), torch.cat((J12, zeros), dim=1)), dim=2)

        # Optimized kron replacement: J ⊗ I_2 using repeat_interleave
        J_expanded = J.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        eye_pattern = self._eye2.unsqueeze(0).expand(J.shape[0], -1, -1)
        eye_tiled = eye_pattern.repeat(1, J.shape[1], J.shape[2])
        return J_expanded * eye_tiled


class Att_H(nn.Module):
    """Attention-based Hamiltonian computation for port-Hamiltonian systems."""
    
    def __init__(self, input_dim, output_dim, hidden_dim, na, device):
        super().__init__()
        self.device = device
        self.activation_soft = nn.Softmax(dim=2)
        self.activation_swish = nn.SiLU()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.na = na

        self.Aq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Ak_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))
        self.Av_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 2 * self.hidden_dim))

        self.Aq_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Ak_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.Av_7 = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))

        self.Bq_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bk_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))
        self.Bv_4 = nn.Parameter(torch.randn(2 * self.hidden_dim, 1))

        self.Bq_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bk_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))
        self.Bv_7 = nn.Parameter(torch.randn(self.hidden_dim, 1))

        self.mlp_in = MLP(input_dim, [2 * hidden_dim]).to(device)
        self.mlp_hidden_4 = MLP(2 * hidden_dim, [hidden_dim]).to(device)
        self.mlp_out = MLP(hidden_dim, [output_dim]).to(device)
        
        # Cache ones tensor for kron replacement
        self.register_buffer('_ones2', torch.ones(1, 2, device=device))

    def forward(self, x, na):
        self.na = na
        x = self.mlp_in(x).unsqueeze(dim=1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bq_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bk_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bv_4.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_hidden_4(x.reshape(-1, 2 * self.hidden_dim)).unsqueeze(dim=1)

        Q = self.activation_swish(
            torch.bmm(self.Aq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bq_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))
        K = self.activation_swish(
            torch.bmm(self.Ak_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bk_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1)).transpose(1, 2)
        V = self.activation_swish(
            torch.bmm(self.Av_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1), x.transpose(1, 2))
            + self.Bv_7.unsqueeze(dim=0).expand(x.shape[0], -1, -1))

        x = self.activation_swish(torch.bmm(self.activation_soft(torch.bmm(Q, K)).to(torch.float32), V).transpose(1, 2))

        x = self.mlp_out(x.reshape(-1, self.hidden_dim)).unsqueeze(dim=1).transpose(1, 2)

        # Reshape, kronecker and post-processing
        # Optimized: replace torch.kron(a, ones(1,2)) with a.repeat_interleave(2, dim=-1)
        x_sq = x ** 2
        M11 = x_sq[:, 0:5, :].sum(1).repeat_interleave(2, dim=-1)
        M12 = x_sq[:, 5:10, :].sum(1).repeat_interleave(2, dim=-1)
        M21 = x_sq[:, 10:15, :].sum(1).repeat_interleave(2, dim=-1)
        M22 = x_sq[:, 15:20, :].sum(1).repeat_interleave(2, dim=-1)
        Mpp = x_sq[:, 20:25, :].sum(1)
        
        # Optimized matrix construction using diag_embed
        Mupper11 = torch.diag_embed(M11)
        Mupper12 = torch.diag_embed(M12)
        Mupper21 = torch.diag_embed(M21)
        Mupper22 = torch.diag_embed(M22)

        M = torch.cat((torch.cat((Mupper11, Mupper21), dim=1), torch.cat((Mupper12, Mupper22), dim=1)), dim=2)
        q = x[:, :4, :]

        return torch.bmm(q.transpose(1, 2), torch.bmm(M, q)).sum(2) + Mpp.sum(1).unsqueeze(1)
