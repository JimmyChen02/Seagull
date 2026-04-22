from typing import Tuple, Optional

import torch
from einops import rearrange

from seagull.nn.modules.linear import Linear
from seagull.nn.modules.module import Module
from seagull.nn.modules.utils.activations import softmax
from seagull.nn.transformer.rope import RotaryPositionalEmbedding, apply_rope


class MultiHeadAttention(Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        max_positions: int = 512,
        num_heads: int = 12,
        use_rope: bool = True,
        base: int = 10000,
        attn_dropout_proba: float = 0.1,
        dropout_proba: float = 0.1,
        causal: bool = True,
        numerically_stable_softmax: bool = False,
    ):
        """
        Initializes the multihead attention that allows the model to jointly attend to information
        from different representation subspaces as described in the paper:
        `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

        Parameters
        ----------
        embedding_dim : int
            Total dimension of the model
        max_position : int
            Number of max position of the attention
        num_heads : int
            Number of parallel attention heads
        use_rope : bool
            If set to False, the model will not use rotary positional embedding
        base : int
            Number of base for the rotary positional embedding
        attn_dropout_proba : float
            Dropout probability on masked attention probability
        dropout_proba : float
            Dropout probability on attention output weights
        causal : bool
            If specified, applies a causal mask as attention mask.
        numerically_stable_softmax : bool
            TO BE FILLED
        """
        super().__init__()

        assert embedding_dim % num_heads == 0, f"embedding_dim: {embedding_dim} not divisible by num_heads: {num_heads}"

        self.embedding_dim = embedding_dim
        self.max_positions = max_positions
        self.num_heads = num_heads
        self.attn_dropout_proba = attn_dropout_proba
        self.numerically_stable_softmax = numerically_stable_softmax

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                head_dim=(embedding_dim // num_heads), max_positions=max_positions, base=base
            )

        causal_mask = torch.ones((max_positions, max_positions), dtype=torch.bool).triu(1) if causal else None
        self.register_buffer("causal_mask", causal_mask)

        # Fused QKV: Using a single transform launches a single (GPU) kernel, which saves us the kernel launch
        # overhead; if we used three `nn.Linear` layers instead, the overhead of launching three kernels would
        # exceed using `torch.split` with one kernel and (3 x embedding_dim) size. Also, `torch.split` doesn't
        # launch any kernels, it returns views of the same underlying memory.
        self.qkv_transform = Linear(in_features=embedding_dim, out_features=(3 * embedding_dim))
        self.output_transform = Linear(
            in_features=embedding_dim, out_features=embedding_dim, dropout_proba=dropout_proba
        )

        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    @staticmethod
    def _make_numerically_stable(similarities: torch.Tensor, softmax_dim: int) -> torch.Tensor:
        return similarities - similarities.amax(dim=softmax_dim, keepdim=True).detach()

    def _update_cache(self, tensor: torch.Tensor, cache: Optional[torch.Tensor]) -> torch.Tensor:
        # key, value: (batch_size, num_heads, max_length, head_dim)
        if cache is None or tensor.shape[-2] == self.max_positions:
            cache = tensor
        else:
            cache = torch.cat([cache, tensor], dim=-2)

        # Use the latest context, when longer than maximum sequence length.
        if cache.shape[-2] > self.max_positions:
            cache = cache[..., -self.max_positions :, :]
        return cache

    def _apply_causal_mask(self, similarities: torch.Tensor) -> torch.Tensor:
        neg_inf = -torch.finfo(similarities.dtype).max
        if self.causal_mask is not None:
            query_seq_length, value_seq_length = similarities.shape[-2], similarities.shape[-1]
            if query_seq_length != 1:  # ignore when kv-cache in effect
                similarities.masked_fill_(self.causal_mask[:query_seq_length, :value_seq_length], neg_inf)
        return similarities

    def masked_attention_probs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Outputs the masked attention probability of the given query and key

        Parameters
        ----------
        query: torch.Tensor
            Tensor representing the query embedding to the MHA model. Queries are compared
            against key-value pairs to produce the output. See “Attention Is All You Need”
            for more details.
        key: torch.Tensor
            Tensor representing the key embedding
        padding_mask: torch.Tensor
            If specified, will fill the similarity tensor with the given value where `mask` is
            set `True`.
        """
        head_dim = query.shape[-1]

        similarities = torch.einsum("bhqd, bhkd -> bhqk", query, key) / (head_dim ** 0.5)

        if padding_mask is not None:
            similarities = similarities.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        similarities = self._apply_causal_mask(similarities)

        if self.numerically_stable_softmax:
            similarities = self._make_numerically_stable(similarities, softmax_dim=-1)

        return softmax(similarities, dim=-1,
                      dropout_proba=self.attn_dropout_proba,
                      training=self.training)

    @staticmethod
    def attention(masked_attn_probs: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # similarities: (batch_size, num_heads, max_length, max_length)
        # value: (batch_size, num_heads, max_length, head_dim)
        # TODO-4.2
        return torch.einsum("bhqk, bhkd -> bhqd", masked_attn_probs, value)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward passes the given embedding through the MHA model.

        Parameters
        ----------
        input_embedding : torch.Tensor
            Tensor representing the input embedding to the MHA model
        padding_mask: Optional[torch.Tensor]
            If specified, a mask indicating which elements within key to ignore
            for the purpose of attention (i.e. treat as “padding”).
        use_kv_cache: bool
        """
        # query, key, value: (batch_size, max_length, embedding_dim)
        query, key, value = self.qkv_transform(input_embeddings).split(split_size=self.embedding_dim, dim=-1)
        # b: batch_size, l: max_length, h: num_heads, d: head_dim
        query, key, value = map(
            lambda tensor: rearrange(tensor, "b l (h d) -> b h l d", h=self.num_heads), (query, key, value)
        )
        if self.use_rope:
            # Handle KV-cache positional embedding by explicitly passing the token position ID.
            start_pos = 0 if (not use_kv_cache or self.k_cache is None) else self.k_cache.shape[-2]
            start_pos = self.max_positions - 1 if start_pos >= self.max_positions else start_pos
            cos_vals, sin_vals = self.rope(
                seq_length=input_embeddings.shape[1], start_pos=start_pos, device=input_embeddings.device
            )
            query, key = apply_rope(query=query, key=key, cos_vals=cos_vals, sin_vals=sin_vals)

        if use_kv_cache:
            self.k_cache = self._update_cache(tensor=key, cache=self.k_cache)
            self.v_cache = self._update_cache(tensor=value, cache=self.v_cache)
            key, value = self.k_cache, self.v_cache

        masked_attn_probs = self.masked_attention_probs(query=query, key=key, padding_mask=padding_mask)
        # attn_out: (batch_size, num_heads, max_length, head_dim) -> (batch_size, max_length, embedding_dim)
        attn_out = self.attention(masked_attn_probs=masked_attn_probs, value=value)
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)", h=self.num_heads)

        output = self.output_transform(attn_out)
        return output, masked_attn_probs
