import torch
import torch.nn as nn
import torch.nn.functional as F

from openfold.np.residue_constants import restypes_with_x


class ESM2PseudoPerplexity(nn.Module):
    """Differentiable ESM2 pseudo-NLL over soft amino-acid sequences."""

    def __init__(
        self,
        model_name: str,
        positions_per_sequence: int = 2,
        max_sequences_per_batch: int = 2,
    ):
        super().__init__()

        try:
            import esm
        except ImportError as error:
            raise ImportError(
                "ESM2 feedback requires fair-esm; install the project's environment.yml"
            ) from error

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if positions_per_sequence < 1 or max_sequences_per_batch < 1:
            raise ValueError("ESM2 scoring limits must be positive")

        self.register_buffer(
            "esm_aa_indices",
            torch.tensor(
                [self.alphabet.get_idx(aa) for aa in restypes_with_x],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.positions_per_sequence = positions_per_sequence
        self.max_sequences_per_batch = max_sequences_per_batch
        self.padding_idx = self.alphabet.padding_idx
        self.mask_idx = self.alphabet.mask_idx
        self.cls_idx = self.alphabet.cls_idx
        self.eos_idx = self.alphabet.eos_idx

    def forward(
        self,
        probabilities: torch.Tensor,
        valid_mask: torch.Tensor,
        score_mask: torch.Tensor,
        randomize_positions: bool = True,
    ) -> torch.Tensor:
        """Return mean masked pseudo-NLL on sampled score_mask positions."""
        batch_size, sequence_length, vocab_size = probabilities.shape
        if vocab_size != len(restypes_with_x):
            raise ValueError(
                f"Expected {len(restypes_with_x)} amino-acid channels, got {vocab_size}"
            )

        valid_mask = valid_mask.bool()
        score_mask = score_mask.bool() & valid_mask
        selected_batches = []
        selected_positions = []
        eligible_batches = torch.nonzero(score_mask.any(dim=-1), as_tuple=False).flatten()
        if eligible_batches.numel() > self.max_sequences_per_batch:
            order = torch.randperm(eligible_batches.numel(), device=eligible_batches.device)
            eligible_batches = eligible_batches[order[:self.max_sequences_per_batch]]

        for batch_index_tensor in eligible_batches:
            batch_index = int(batch_index_tensor.item())
            candidates = torch.nonzero(score_mask[batch_index], as_tuple=False).flatten()
            if candidates.numel() == 0:
                continue
            count = min(self.positions_per_sequence, candidates.numel())
            if randomize_positions:
                order = torch.randperm(candidates.numel(), device=candidates.device)[:count]
            else:
                order = torch.linspace(
                    0,
                    candidates.numel() - 1,
                    count,
                    device=candidates.device,
                ).round().long()
            selected_batches.append(
                torch.full((count,), batch_index, device=candidates.device, dtype=torch.long)
            )
            selected_positions.append(candidates[order])

        if not selected_batches:
            return probabilities.sum() * 0.0

        batch_indices = torch.cat(selected_batches)
        positions = torch.cat(selected_positions)
        selected_probabilities = probabilities.index_select(0, batch_indices)
        selected_valid = valid_mask.index_select(0, batch_indices)

        embedding = self.model.embed_tokens.weight
        aa_embedding = embedding.index_select(0, self.esm_aa_indices)
        residue_embeddings = selected_probabilities.to(embedding.dtype) @ aa_embedding
        residue_embeddings = residue_embeddings * selected_valid.unsqueeze(-1)

        row_count = batch_indices.numel()
        tokens = torch.full(
            (row_count, sequence_length + 2),
            self.padding_idx,
            dtype=torch.long,
            device=probabilities.device,
        )
        tokens[:, 0] = self.cls_idx
        residue_tokens = tokens[:, 1:sequence_length + 1]
        residue_tokens.fill_(self.padding_idx)
        residue_tokens[selected_valid] = self.esm_aa_indices[0]
        lengths = selected_valid.sum(dim=-1)
        row_indices = torch.arange(row_count, device=probabilities.device)
        tokens[row_indices, lengths + 1] = self.eos_idx
        tokens[row_indices, positions + 1] = self.mask_idx

        x = embedding.new_zeros((row_count, sequence_length + 2, embedding.shape[-1]))
        x[:, 0] = embedding[self.cls_idx]
        x[:, 1:sequence_length + 1] = residue_embeddings
        x[row_indices, lengths + 1] = embedding[self.eos_idx]
        x[row_indices, positions + 1] = embedding[self.mask_idx]
        x = self.model.embed_scale * x

        padding_mask = tokens.eq(self.padding_idx)
        if self.model.token_dropout:
            x = x.masked_fill(tokens.eq(self.mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8
            mask_ratio_observed = tokens.eq(self.mask_idx).sum(-1).to(x.dtype)
            mask_ratio_observed = mask_ratio_observed / (~padding_mask).sum(-1).to(x.dtype)
            x = x * ((1 - mask_ratio_train) / (1 - mask_ratio_observed))[:, None, None]

        x = x * (~padding_mask).unsqueeze(-1).to(x.dtype)
        positional_embedding = getattr(self.model, "embed_positions", None)
        if positional_embedding is not None:
            x = x + positional_embedding(tokens)
        embedding_norm = getattr(self.model, "emb_layer_norm_before", None)
        if embedding_norm is not None:
            x = embedding_norm(x)
        dropout = getattr(self.model, "dropout_module", None)
        if dropout is not None:
            x = dropout(x)

        x = x.transpose(0, 1)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.model.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
        x = self.model.emb_layer_norm_after(x).transpose(0, 1)
        logits = self.model.lm_head(x)[row_indices, positions + 1]
        log_probabilities = F.log_softmax(logits, dim=-1).index_select(
            -1, self.esm_aa_indices
        )
        targets = probabilities[batch_indices, positions].to(log_probabilities.dtype)
        return -(targets * log_probabilities).sum(-1).mean()

    def guide(
        self,
        probabilities: torch.Tensor,
        valid_mask: torch.Tensor,
        active_mask: torch.Tensor,
        steps: int,
        step_size: float,
    ) -> torch.Tensor:
        """Refine sampled probabilities by descending the ESM2 pseudo-NLL."""
        from .flow_utils import simplex_proj

        current = probabilities.detach()
        active_mask = active_mask.bool() & valid_mask.bool()
        with torch.enable_grad():
            for _ in range(steps):
                candidate = current.detach().requires_grad_(True)
                loss = self(
                    candidate,
                    valid_mask,
                    active_mask,
                    randomize_positions=False,
                )
                gradient = torch.autograd.grad(loss, candidate)[0]
                gradient = gradient * active_mask.unsqueeze(-1)
                gradient = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                updated = simplex_proj(candidate - step_size * gradient)
                current = torch.where(active_mask.unsqueeze(-1), updated, current).detach()
        return current