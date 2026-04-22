from typing import Union, Literal, Optional, Tuple, List

import torch
from torch import nn

from seagull.model.components.embedding import Embedding
from seagull.model.components.transformer_layer import TransformerLayer
from seagull.nn.modules.module import Module
from seagull.nn.modules.rms_norm import RMSNorm


class Seagull(Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        padding_idx: Optional[int] = None,
        layer_norm_embedding: bool = False,
        num_layers: int = 12,
        intermediate_dim: int = 2048,
        max_positions: int = 512,
        dropout_proba: float = 0.1,
        num_heads: int = 12,
        use_rope: bool = True,
        base: int = 10000,
        attn_dropout_proba: float = 0.1,
        causal: bool = True,
        ffn_bias: bool = False,
        ffn_activation: str = "swish",
        layer_norm_mode: Literal["pre", "post"] = "pre",
        layer_norm_type: str = "rms",
    ):
        """
        A transformer model consisting of one or more :py:class:`seagull.model.components.transformer_layer.TransformerLayer` transformer layers.

        Parameters
        ----------
        vocab_size : int
            The size of the vocabulary used.
        embedding_dim : int
            The embedding dimension of the inputs.
        padding_idx : int
            The token index corresponding to padding tokens; the padded token embedding is a vector of all zeros.
        layer_norm_embedding : bool
            If set to False, applies layer normalization for embeddings and otherwise defaults to `layer_norm_type`.
        num_layers : int
            The number of transformer layers to include in the model.
        intermediate_dim : int
            The intermediate dimension of the feed-forward layer.
        max_positions : int
            Number of max positions of the attention computation.
        dropout_proba : float
            The dropout probability used for the feed-forward network and attention computation.
        num_heads : int
            Number of parallel attention heads.
        use_rope : bool
            If set to False, the attention computation will not use rotary positional embedding.
        base : int
            Number of bases for the rotary positional embedding in the attention computation.
        attn_dropout_proba : float
            The dropout probability on the masked attention probability.
        causal : bool
            If specified, applies a causal mask as attention mask.
        ffn_bias : bool
             Whether to include a bias for the layers in the feed-forward network.
        ffn_activation : str
            The type of activation function to use for the layers in the feed-forward network.
        layer_norm_mode : Literal["pre", "post"]
            The layer normalization strategy to use, can be "pre" or "post".
        layer_norm_type : str
            The type of layer normalization to apply, either root mean square or layer normalization.
        """
        super().__init__()

        self.num_layers = num_layers
        self._dropout_proba = dropout_proba

        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_positions=max_positions,
            padding_idx=padding_idx,
            use_rope=use_rope,
            layer_norm_type=(layer_norm_type if layer_norm_embedding else None),
            dropout_proba=dropout_proba,
        )
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    embedding_dim=embedding_dim,
                    intermediate_dim=intermediate_dim,
                    max_positions=max_positions,
                    dropout_proba=dropout_proba,
                    num_heads=num_heads,
                    use_rope=use_rope,
                    base=base,
                    attn_dropout_proba=attn_dropout_proba,
                    causal=causal,
                    ffn_bias=ffn_bias,
                    ffn_activation=ffn_activation,
                    layer_norm_mode=layer_norm_mode,
                    layer_norm_type=layer_norm_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = self._get_layer_norm(layer_norm_type=layer_norm_type)

    def reset_kv_cache(self):
        for layer in self.transformer_layers:
            layer.reset_kv_cache()

    def _get_layer_norm(self, layer_norm_type: str) -> Union[Module, nn.Module]:
        if layer_norm_type.startswith("rms"):
            return RMSNorm(dimension=self.embedding.embedding_dim, eps=1e-8, dropout_proba=self._dropout_proba)
        else:
            return nn.Sequential(
                nn.LayerNorm(normalized_shape=self.embedding.embedding_dim, eps=1e-8),
                nn.Dropout(p=self._dropout_proba),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
        return_output_at_all_layers: bool = True,
        return_attentions: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        Computes a forward pass through each of the transformer layers using the given inputs.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input IDs passed into the transformer.
        position_ids : Optional[torch.Tensor]
            Optional position IDs to use for positional embeddings.
        padding_mask : Optional[torch.Tensor]
            Optional padding mask indicating which tokens in the embeddings, if any, are padding tokens.
        use_kv_cache : bool
            Whether to use cached keys and values for attention.
        return_output_at_all_layers : bool
            Whether to return the output at all transformer layers, or only the final layer.
        return_attentions : bool
            Whether to return the attentions of all transformer layers in addition to the outputs.

        Returns
        -------
        Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor]]]
            A tuple of the outputs from the forward pass and attention probabilities from each layer, or the outputs
            alone, depending on `return_attentions`. Further, the outputs will consist of values from the final
            transformer layer or all layers, depending on `return_output_at_all_layers`.
        """
        if not use_kv_cache and padding_mask is None and self.embedding.padding_idx is not None:
            padding_mask = (input_ids == self.embedding.padding_idx).to(input_ids.device)

        all_outputs, all_attns = [], []
        output = self.embedding(input_ids=input_ids, position_ids=position_ids)
        for layer_num, layer in enumerate(self.transformer_layers):
            output, masked_attn_probs = layer(
                input_embeddings=output, padding_mask=padding_mask, use_kv_cache=use_kv_cache
            )
            if layer_num == self.num_layers - 1 and layer.layer_norm_mode == "pre":
                output = self.layer_norm(output)  # apply layer norm, if the final output

            if return_output_at_all_layers or layer_num == self.num_layers - 1:
                all_outputs.append(output)
            if return_attentions:
                all_attns.append(masked_attn_probs)

        return (all_outputs, all_attns) if return_attentions else all_outputs