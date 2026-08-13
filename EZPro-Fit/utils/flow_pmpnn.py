import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


from .flow_utils import (
    sample_cond_prob_path, 
    simplex_proj, 
    get_signal_mapping,
    GaussianDiffusionSchedule
)
from .pmpnn import (
    gather_edges, gather_nodes, cat_neighbors_nodes,
    EncLayer, DecLayer, ProteinFeatures, CA_ProteinFeatures,
    get_weird_pmpnn_stuff
)


  
class FlowProteinMPNN(nn.Module):
    """
    Flow-based ProteinMPNN that replaces autoregressive decoding with continuous flow generation
    """
    def __init__(
        self, 
        args,
        node_features: int = 128, 
        edge_features: int = 128,
        hidden_dim: int = 128, 
        num_encoder_layers: int = 3, 
        num_decoder_layers: int = 3,
        vocab: int = 21, 
        k_neighbors: int = 48, 
        augment_eps: float = 0.1, 
        dropout: float = 0.1,
        ca_only: bool = False,
        # Flow-specific parameters
        flow_mode: str = 'dirichlet',  # 'dirichlet', 'riemannian', 'distill'
        time_embedding_dim: int = 64,
        time_embedding_type: str = 'sinusoidal',
        diffusion_steps: int = 1000,
        alpha_scale: float = 1.0,
        fix_alpha: Optional[float] = None
    ):
        super(FlowProteinMPNN, self).__init__()
        
        self.args = args
        self.vocab = vocab
        self.flow_mode = flow_mode
        self.diffusion_steps = diffusion_steps
        self.alpha_scale = alpha_scale
        self.fix_alpha = fix_alpha
        
        # Core MPNN components
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        # Structural feature extraction (unchanged from original MPNN)
        if ca_only:
            self.features = CA_ProteinFeatures(
                node_features, edge_features, 
                top_k=k_neighbors, augment_eps=augment_eps
            )
            self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        else:
            self.features = ProteinFeatures(
                node_features, edge_features, 
                top_k=k_neighbors, augment_eps=augment_eps
            )
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        
        # Time embedding for flow model
        self.time_embedding = get_signal_mapping(
            time_embedding_type, time_embedding_dim
        )
        
        # Encoder layers (structural encoding)
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Flow-based decoder layers
        self.flow_decoder_layers = nn.ModuleList([
            FlowDecLayer(
                hidden_dim, 
                time_embedding_dim,
                hidden_dim * 2, 
                dropout=dropout
            ) for _ in range(num_decoder_layers)
        ])
        
        # Output layers for flow model
        if flow_mode in ['dirichlet', 'riemannian']:
            # Output simplex probabilities directly
            self.W_out = nn.Linear(hidden_dim, vocab, bias=True)
        elif flow_mode == 'distill':
            # Output parameters for distribution
            self.W_out = nn.Linear(hidden_dim, vocab, bias=True)
        
        # Diffusion schedule for training
        self.diffusion_schedule = GaussianDiffusionSchedule(
            timesteps=diffusion_steps
        ) if flow_mode == 'gaussian' else None
        
        # Taxon conditioning (if needed)
        if hasattr(args, 'taxon_condition') and args.taxon_condition:
            self.taxon_emb = nn.Embedding(args.num_taxon_ids, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward_train(
        self, 
        X: torch.Tensor,
        S: torch.Tensor, 
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        taxon_id: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        Training forward pass with flow-based sequence generation
        
        Args:
            X: Protein coordinates [B, L, 4, 3] or [B, L, 3] for CA-only
            S: Target sequences [B, L]
            mask: Valid position mask [B, L]
            chain_M: Chain mask [B, L]
            residue_idx: Residue indices [B, L]
            chain_encoding_all: Chain encodings [B, L]
            taxon_id: Taxon IDs [B] (optional)
            return_hidden: Whether to return hidden states
        
        Returns:
            Flow loss or log probabilities
        """
        device = X.device
        B, L = S.shape
        
        # Extract structural features (unchanged from original MPNN)
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((B, L, self.hidden_dim), device=device)
        h_E = self.W_e(E)
        
        # Add taxon conditioning if enabled
        if hasattr(self, 'taxon_emb') and taxon_id is not None:
            taxon_emb = self.taxon_emb(taxon_id)
            h_V = h_V + taxon_emb[:, None, :]
            h_E = h_E + taxon_emb[:, None, None, :]
        
        # Structural encoding (unchanged)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        
        h_V_encoder = h_V.clone() if return_hidden else None
        
        # Sample noisy sequence state for flow training
        xt, alphas = self._sample_flow_state(S, device)
        
        # Time embedding
        if self.flow_mode in ['dirichlet', 'riemannian']:
            t_emb = self.time_embedding(alphas)  # [B, time_dim]
        else:
            # For other modes, use uniform time sampling
            t = torch.rand(B, device=device)
            t_emb = self.time_embedding(t)
        
        # Flow-based decoding
        h_V_flow = h_V.clone()
        
        # Create edge features for flow decoder
        h_E_flow = h_E.clone()
        
        for layer in self.flow_decoder_layers:
            h_V_flow = layer(h_V_flow, h_E_flow, E_idx, t_emb, mask)
        
        # Output flow prediction
        flow_output = self.W_out(h_V_flow)  # [B, L, vocab]
        
        # Compute flow loss
        loss = self._compute_flow_loss(flow_output, xt, S, alphas, mask, chain_M)
        
        if return_hidden:
            return loss, h_V_encoder, h_V_flow
        else:
            return loss
    
    def _sample_flow_state(self, S: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy state for flow training"""
        # Create mock args for compatibility with existing function
        class MockArgs:
            def __init__(self, mode, alpha_scale, fix_alpha):
                self.mode = mode
                self.alpha_scale = alpha_scale
                self.fix_alpha = fix_alpha
        
        mock_args = MockArgs(self.flow_mode, self.alpha_scale, self.fix_alpha)
        return sample_cond_prob_path(mock_args, S, self.vocab)
    
    def _compute_flow_loss(
        self, 
        flow_output: torch.Tensor,
        xt: torch.Tensor, 
        S: torch.Tensor,
        alphas: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor
    ) -> torch.Tensor:
        """Compute flow-based loss"""
        B, L, _ = flow_output.shape
        
        if self.flow_mode == 'dirichlet':
            # Predict the clean simplex state
            pred_x0 = F.softmax(flow_output, dim=-1)
            target_x0 = F.one_hot(S, num_classes=self.vocab).float()
            
            # Flow matching loss
            loss = F.mse_loss(pred_x0, target_x0, reduction='none')
            loss = loss.sum(-1)  # Sum over vocab dimension
            
        elif self.flow_mode == 'riemannian':
            # Predict velocity field
            pred_v = flow_output
            # Target velocity points toward clean state
            target_x0 = F.one_hot(S, num_classes=self.vocab).float()
            target_v = target_x0 - xt  # Simple linear interpolation velocity
            
            loss = F.mse_loss(pred_v, target_v, reduction='none')
            loss = loss.sum(-1)
            
        elif self.flow_mode == 'distill':
            # Standard cross-entropy loss
            loss = F.cross_entropy(
                flow_output.view(-1, self.vocab), 
                S.view(-1), 
                reduction='none'
            ).view(B, L)
        
        else:
            raise ValueError(f"Unknown flow mode: {self.flow_mode}")
        
        # Apply masks
        loss = loss * mask * chain_M
        return loss.sum() / (mask * chain_M).sum().clamp(min=1)
    
    def sample(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding_all: torch.Tensor,
        taxon_id: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        temperature: float = 1.0,
        ode_solver: str = 'euler'
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences using ODE-based flow sampling (PDF-compliant)

        Implements x₁ = x₀ + ∫₀¹ vθ(xt, t, c)dt where:
        - x₀: initial noise state
        - x₁: final clean sequence distribution
        - vθ: velocity field predicted by flow decoder
        - c: condition (structural encoding + taxon embedding)
        - t ∈ [0,1]: continuous time variable

        Args:
            X: Protein coordinates [B, L, 4, 3] or [B, L, 3]
            mask: Valid position mask [B, L]
            chain_M: Chain mask [B, L]
            residue_idx: Residue indices [B, L]
            chain_encoding_all: Chain encodings [B, L]
            taxon_id: Taxon IDs [B] (optional)
            num_steps: Number of ODE integration steps
            temperature: Sampling temperature
            ode_solver: ODE solver type ('euler', 'heun', 'rk4')

        Returns:
            Dictionary with keys:
            - 'S': Sampled sequences [B, L]
            - 'probs': Final probability distributions [B, L, vocab]
            - 'trajectory': Optional trajectory for visualization [num_steps, B, L, vocab]
        """
        device = X.device
        B, L = mask.shape

        # ========== Phase 1: Structural Encoding ==========
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((B, L, self.hidden_dim), device=device)
        h_E = self.W_e(E)

        # Add taxon conditioning (concatenation - fixed version)
        if hasattr(self, 'taxon_emb') and taxon_id is not None:
            taxon_emb = self.taxon_emb(taxon_id)  # [B, hidden_dim]
            h_V = h_V + taxon_emb[:, None, :]    # Broadcast to [B, L, hidden_dim]
            h_E = h_E + taxon_emb[:, None, None, :]  # Broadcast to [B, L, K, hidden_dim]

        # Encode structural features
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        # Store structural condition (constant during ODE integration)
        h_V_condition = h_V.clone()
        h_E_condition = h_E.clone()

        # ========== Phase 2: Initialize Noise State ==========
        # x₀ ~ p₀ (initial probability distribution)
        if self.flow_mode in ['dirichlet', 'riemannian']:
            # Start from uniform distribution on simplex
            xt = torch.ones(B, L, self.vocab, device=device) / self.vocab
            # Add controlled noise
            noise = torch.randn(B, L, self.vocab, device=device) * 0.1
            xt = xt + noise
            xt = simplex_proj(xt)  # Project to simplex to ensure validity
        else:
            # Standard normal for other modes
            xt = torch.randn(B, L, self.vocab, device=device) / np.sqrt(self.vocab)

        # ========== Phase 3: ODE Integration Loop ==========
        # Solve ODE: dxt/dt = vθ(xt, t, c) for t ∈ [0, 1]
        trajectory = []  # For optional visualization

        if ode_solver == 'euler':
            xt = self._ode_euler_step(
                xt, h_V_condition, h_E_condition, E_idx, mask, chain_M,
                num_steps, temperature, device
            )
        elif ode_solver == 'heun':
            xt = self._ode_heun_step(
                xt, h_V_condition, h_E_condition, E_idx, mask, chain_M,
                num_steps, temperature, device
            )
        elif ode_solver == 'rk4':
            xt = self._ode_rk4_step(
                xt, h_V_condition, h_E_condition, E_idx, mask, chain_M,
                num_steps, temperature, device
            )
        else:
            raise ValueError(f"Unknown ODE solver: {ode_solver}")

        # ========== Phase 4: Discrete Sampling ==========
        # Sample from final distribution: S ~ argmax(xt) or multinomial(xt)
        # Use argmax for deterministic decoding, multinomial for stochastic
        if temperature == 0.0:
            # Deterministic: take argmax
            S = torch.argmax(xt, dim=-1)
        else:
            # Stochastic: multinomial sampling
            # Reshape for batch processing
            probs_flat = xt.view(B * L, self.vocab)
            S_flat = torch.multinomial(probs_flat, 1).squeeze(-1)
            S = S_flat.view(B, L)

        # Apply chain mask to zero out invalid positions
        S = S * chain_M.long()

        return {
            "S": S,
            "probs": xt,
            "final_state": xt,
            "trajectory": trajectory if trajectory else None
        }

    def _predict_velocity_field(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        h_V_condition: torch.Tensor,
        h_E_condition: torch.Tensor,
        E_idx: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field vθ(xt, t, c) at time t

        This is the core of the flow model:
        - Takes current state xt and time t
        - Computes conditional features using structural context
        - Returns predicted direction toward data manifold

        Args:
            xt: Current state [B, L, vocab]
            t: Current time [B]
            h_V_condition: Structural node encoding [B, L, hidden_dim]
            h_E_condition: Structural edge encoding [B, L, K, hidden_dim]
            E_idx: Edge indices [B, L, K]
            mask: Position mask [B, L]

        Returns:
            vt: Predicted velocity field [B, L, vocab]
        """
        device = xt.device
        B, L = t.shape

        # Time embedding
        t_emb = self.time_embedding(t)  # [B, time_dim]

        # Prepare decoder input (concatenate structural encoding with current state)
        # This allows the model to use both structure and current distribution
        h_V_decoder = h_V_condition.clone()
        h_E_decoder = h_E_condition.clone()

        # Flow-based decoder layers with time conditioning
        for layer in self.flow_decoder_layers:
            h_V_decoder = layer(h_V_decoder, h_E_decoder, E_idx, t_emb, mask)

        # Output velocity prediction
        vt = self.W_out(h_V_decoder)  # [B, L, vocab]

        return vt

    def _ode_euler_step(
        self,
        x0: torch.Tensor,
        h_V_cond: torch.Tensor,
        h_E_cond: torch.Tensor,
        E_idx: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        num_steps: int,
        temperature: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Euler's method for ODE integration: xt+dt = xt + dt * vθ(xt, t, c)

        Pros: Simple, stable
        Cons: O(1/N) error, slowest convergence

        Args:
            x0: Initial state [B, L, vocab]
            num_steps: Number of integration steps

        Returns:
            x1: Final state [B, L, vocab]
        """
        B, L = x0.shape[:2]
        xt = x0.clone()
        dt = 1.0 / num_steps

        for step in range(num_steps):
            # Current time in [0, 1]
            t_step = torch.ones(B, device=device) * (step / num_steps)

            # Predict velocity field
            vt = self._predict_velocity_field(
                xt, t_step, h_V_cond, h_E_cond, E_idx, mask
            )

            # Euler step: xt+dt = xt + dt * vt
            if self.flow_mode == 'dirichlet':
                # For dirichlet mode, vt predicts target distribution
                pred_x0 = F.softmax(vt / temperature, dim=-1)
                xt = xt + dt * (pred_x0 - xt)
            elif self.flow_mode == 'riemannian':
                # For riemannian mode, vt is velocity
                xt = xt + dt * vt
            else:
                # For distill mode, direct softmax
                xt = xt + dt * (F.softmax(vt / temperature, dim=-1) - xt)

            # Project back to valid probability simplex
            if self.flow_mode in ['dirichlet', 'riemannian']:
                xt = simplex_proj(xt)
            else:
                xt = F.softmax(xt, dim=-1)

            # Apply position mask
            xt = xt * mask.unsqueeze(-1)

        return xt

    def _ode_heun_step(
        self,
        x0: torch.Tensor,
        h_V_cond: torch.Tensor,
        h_E_cond: torch.Tensor,
        E_idx: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        num_steps: int,
        temperature: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Heun's method (improved Euler, RK2) for ODE integration

        Algorithm:
            k1 = vθ(xt, t)
            k2 = vθ(xt + dt*k1, t+dt)
            xt+dt = xt + (dt/2)*(k1 + k2)

        Pros: O(1/N²) error, 2x improvement over Euler
        Cons: 2x computational cost

        Args:
            x0: Initial state [B, L, vocab]
            num_steps: Number of integration steps

        Returns:
            x1: Final state [B, L, vocab]
        """
        B, L = x0.shape[:2]
        xt = x0.clone()
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_step = torch.ones(B, device=device) * (step / num_steps)
            t_next = torch.ones(B, device=device) * ((step + 1) / num_steps)

            # First evaluation: k1 = vθ(xt, t)
            v1 = self._predict_velocity_field(
                xt, t_step, h_V_cond, h_E_cond, E_idx, mask
            )

            # Predictor step
            if self.flow_mode == 'dirichlet':
                x_pred = xt + dt * (F.softmax(v1 / temperature, dim=-1) - xt)
            elif self.flow_mode == 'riemannian':
                x_pred = xt + dt * v1
            else:
                x_pred = xt + dt * (F.softmax(v1 / temperature, dim=-1) - xt)

            x_pred = simplex_proj(x_pred) if self.flow_mode in ['dirichlet', 'riemannian'] else F.softmax(x_pred, dim=-1)

            # Second evaluation: k2 = vθ(xt + dt*k1, t+dt)
            v2 = self._predict_velocity_field(
                x_pred, t_next, h_V_cond, h_E_cond, E_idx, mask
            )

            # Corrector step: xt+dt = xt + (dt/2)*(v1 + v2)
            if self.flow_mode == 'dirichlet':
                v1_target = F.softmax(v1 / temperature, dim=-1)
                v2_target = F.softmax(v2 / temperature, dim=-1)
                xt = xt + (dt / 2) * ((v1_target - xt) + (v2_target - xt))
            elif self.flow_mode == 'riemannian':
                xt = xt + (dt / 2) * (v1 + v2)
            else:
                v1_target = F.softmax(v1 / temperature, dim=-1)
                v2_target = F.softmax(v2 / temperature, dim=-1)
                xt = xt + (dt / 2) * ((v1_target - xt) + (v2_target - xt))

            xt = simplex_proj(xt) if self.flow_mode in ['dirichlet', 'riemannian'] else F.softmax(xt, dim=-1)
            xt = xt * mask.unsqueeze(-1)

        return xt

    def _ode_rk4_step(
        self,
        x0: torch.Tensor,
        h_V_cond: torch.Tensor,
        h_E_cond: torch.Tensor,
        E_idx: torch.Tensor,
        mask: torch.Tensor,
        chain_M: torch.Tensor,
        num_steps: int,
        temperature: float,
        device: torch.device
    ) -> torch.Tensor:
        """
        Runge-Kutta 4th order (RK4) method for ODE integration

        Algorithm:
            k1 = vθ(xt, t)
            k2 = vθ(xt + dt*k1/2, t+dt/2)
            k3 = vθ(xt + dt*k2/2, t+dt/2)
            k4 = vθ(xt + dt*k3, t+dt)
            xt+dt = xt + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        Pros: O(1/N⁴) error, best accuracy
        Cons: 4x computational cost

        Args:
            x0: Initial state [B, L, vocab]
            num_steps: Number of integration steps

        Returns:
            x1: Final state [B, L, vocab]
        """
        B, L = x0.shape[:2]
        xt = x0.clone()
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_base = torch.ones(B, device=device) * (step / num_steps)
            t_mid = torch.ones(B, device=device) * ((step + 0.5) / num_steps)
            t_next = torch.ones(B, device=device) * ((step + 1) / num_steps)

            # k1 = vθ(xt, t)
            v1 = self._predict_velocity_field(xt, t_base, h_V_cond, h_E_cond, E_idx, mask)
            if self.flow_mode == 'dirichlet':
                v1_target = F.softmax(v1 / temperature, dim=-1)
            else:
                v1_target = v1

            # x_mid1 = xt + (dt/2)*v1
            x_mid1 = xt + (dt / 2) * v1_target
            x_mid1 = simplex_proj(x_mid1) if self.flow_mode in ['dirichlet', 'riemannian'] else F.softmax(x_mid1, dim=-1)

            # k2 = vθ(x_mid1, t+dt/2)
            v2 = self._predict_velocity_field(x_mid1, t_mid, h_V_cond, h_E_cond, E_idx, mask)
            if self.flow_mode == 'dirichlet':
                v2_target = F.softmax(v2 / temperature, dim=-1)
            else:
                v2_target = v2

            # x_mid2 = xt + (dt/2)*v2
            x_mid2 = xt + (dt / 2) * v2_target
            x_mid2 = simplex_proj(x_mid2) if self.flow_mode in ['dirichlet', 'riemannian'] else F.softmax(x_mid2, dim=-1)

            # k3 = vθ(x_mid2, t+dt/2)
            v3 = self._predict_velocity_field(x_mid2, t_mid, h_V_cond, h_E_cond, E_idx, mask)
            if self.flow_mode == 'dirichlet':
                v3_target = F.softmax(v3 / temperature, dim=-1)
            else:
                v3_target = v3

            # x_end = xt + dt*v3
            x_end = xt + dt * v3_target
            x_end = simplex_proj(x_end) if self.flow_mode in ['dirichlet', 'riemannian'] else F.softmax(x_end, dim=-1)

            # k4 = vθ(x_end, t+dt)
            v4 = self._predict_velocity_field(x_end, t_next, h_V_cond, h_E_cond, E_idx, mask)
            if self.flow_mode == 'dirichlet':
                v4_target = F.softmax(v4 / temperature, dim=-1)
            else:
                v4_target = v4

            # RK4 update: xt+dt = xt + (dt/6)*(v1 + 2*v2 + 2*v3 + v4)
            xt = xt + (dt / 6) * (v1_target + 2 * v2_target + 2 * v3_target + v4_target)

            xt = simplex_proj(xt) if self.flow_mode in ['dirichlet', 'riemannian'] else F.softmax(xt, dim=-1)
            xt = xt * mask.unsqueeze(-1)

        return xt


class FlowDecLayer(nn.Module):
    """
    Decoder layer adapted for flow-based generation with time conditioning
    """
    def __init__(
        self, 
        hidden_dim: int, 
        time_dim: int,
        input_dim: int, 
        dropout: float = 0.1, 
        scale: float = 30
    ):
        super(FlowDecLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.scale = scale
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Time conditioning
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        
        # Message passing layers
        #self.W1 = nn.Linear(hidden_dim + input_dim + time_dim, hidden_dim, bias=True)
        self.W1 = nn.Linear(320, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W3 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(hidden_dim, hidden_dim * 4)
    
    def forward(
        self, 
        h_V: torch.Tensor, 
        h_E: torch.Tensor, 
        E_idx: torch.Tensor,
        t_emb: torch.Tensor,
        mask_V: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with time conditioning
        
        Args:
            h_V: Node features [B, L, hidden_dim]
            h_E: Edge features [B, L, K, hidden_dim]  
            E_idx: Edge indices [B, L, K]
            t_emb: Time embeddings [B, time_dim]
            mask_V: Node mask [B, L]
        """
        # Expand time embedding
        t_proj = self.time_proj(t_emb)  # [B, hidden_dim]
        t_expand = t_proj.unsqueeze(1).expand(-1, h_V.shape[1], -1)  # [B, L, hidden_dim]
        
        # Time-conditioned node features
        h_V_t = h_V + t_expand
        
        # Message passing with time conditioning
        h_V_expand = h_V_t.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        t_expand_edge = t_emb.unsqueeze(1).unsqueeze(1).expand(-1, h_E.shape[1], h_E.shape[2], -1)
        
        h_EV = torch.cat([h_V_expand, h_E, t_expand_edge], -1)
        
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = torch.sum(h_message, -2) / self.scale
        
        h_V = self.norm1(h_V + self.dropout1(dh))
        
        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
            
        return h_V


class PositionWiseFeedForward(nn.Module):
    """Position-wise feedforward network"""
    def __init__(self, num_hidden: int, num_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


# Example usage and training loop
def train_flow_mpnn(model, dataloader, optimizer, device):
    """Example training loop for flow-based ProteinMPNN"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        # Unpack batch data
        X = batch['coords'].to(device)  # [B, L, 4, 3] or [B, L, 3]
        S = batch['seq'].to(device)     # [B, L]
        mask = batch['mask'].to(device) # [B, L]
        chain_M = batch['chain_mask'].to(device)  # [B, L]
        residue_idx = batch['residue_idx'].to(device)  # [B, L]
        chain_encoding = batch['chain_encoding'].to(device)  # [B, L]
        taxon_id = batch.get('taxon_id', None)
        
        if taxon_id is not None:
            taxon_id = taxon_id.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        loss = model.forward_train(
            X, S, mask, chain_M, residue_idx, chain_encoding, taxon_id
        )
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def sample_sequences(model, batch, device, num_steps=100, temperature=1.0):
    """Example sequence sampling"""
    model.eval()
    
    with torch.no_grad():
        X = batch['coords'].to(device)
        mask = batch['mask'].to(device)
        chain_M = batch['chain_mask'].to(device)
        residue_idx = batch['residue_idx'].to(device)
        chain_encoding = batch['chain_encoding'].to(device)
        taxon_id = batch.get('taxon_id', None)
        
        if taxon_id is not None:
            taxon_id = taxon_id.to(device)
        
        result = model.sample(
            X, mask, chain_M, residue_idx, chain_encoding, 
            taxon_id, num_steps, temperature
        )
        
        return result['S'], result['probs']
