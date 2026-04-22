from typing import Union, Literal, Optional, Tuple

import torch
from torch import nn

from seagull.nn.modules.module import Module
from seagull.nn.modules.rms_norm import RMSNorm
from seagull.nn.transformer.ffn import FFN
from seagull.nn.transformer.mha import MultiHeadAttention


class TransformerLayer(Module):
    def __init__(
        self,
        embedding_dim: int = 768,
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
        A single transformer layer including attention and normalization computations.

        Parameters
        ----------
        embedding_dim : int
            The embedding dimension of the inputs.
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

        self.embedding_dim = embedding_dim
        self._dropout_proba = dropout_proba

        self.mha = MultiHeadAttention(
            embedding_dim=embedding_dim,
            max_positions=max_positions,
            num_heads=num_heads,
            use_rope=use_rope,
            base=base,
            attn_dropout_proba=attn_dropout_proba,
            dropout_proba=dropout_proba,
            causal=causal,
        )
        self.ffn = FFN(
            embedding_dim=embedding_dim,
            intermediate_dim=intermediate_dim,
            bias=ffn_bias,
            activation=ffn_activation,
            dropout_proba=dropout_proba,
        )

        self.layer_norm_mode = layer_norm_mode
        self.layer_norm = self._get_layer_norm(layer_norm_type=layer_norm_type)

    def reset_kv_cache(self):
        self.mha.reset_kv_cache()

    def _get_layer_norm(self, layer_norm_type: str) -> Union[Module, nn.Module]:
        if layer_norm_type.startswith("rms"):
            return RMSNorm(dimension=self.embedding_dim, eps=1e-8, dropout_proba=self._dropout_proba)
        else:
            return nn.Sequential(
                nn.LayerNorm(normalized_shape=self.embedding_dim, eps=1e-8), nn.Dropout(p=self._dropout_proba)
            )

    def maybe_apply_layer_norm(
        self, tensor: torch.Tensor, current_layer_norm_application: Literal["pre", "post"]
    ) -> torch.Tensor:
        """
        Applies layer normalization if current layer normalization type matches preset layer norm mode.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to possibly apply layer normalization to.
        current_layer_norm_application : torch.Tensor
            The current layer normalization type.

        Returns
        -------
        torch.Tensor
            Possibly layer normalized version of input tensor.
        """
        if current_layer_norm_application == self.layer_norm_mode:
            return self.layer_norm(tensor)
        return tensor

    def forward(
        self,
        input_embeddings: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        use_kv_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes a forward pass through the transformer on the input embeddings.

        Parameters
        ----------
        input_embeddings : torch.Tensor
            Input tensor of embeddings.
        padding_mask : Optional[torch.Tensor]
            Optional padding mask indicating which tokens in the embeddings, if any, are padding tokens.
        use_kv_cache : bool
            Whether to use cached keys and values for attention.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of transformer layer outputs and masked attention probabilities.
        """
        # TODO-6.1
        x = self.maybe_apply_layer_norm(
            input_embeddings,
            current_layer_norm_application="pre",
        )
        mha_output, masked_attention_probs = self.mha(
            x,
            padding_mask=padding_mask,
            use_kv_cache=use_kv_cache,
        )
        x = input_embeddings + mha_output
        x = self.maybe_apply_layer_norm(
            x,
            current_layer_norm_application="post",
        )

        y = self.maybe_apply_layer_norm(
            x,
            current_layer_norm_application="pre",
        )
        ffn_output = self.ffn(y)
        x = x + ffn_output
        x = self.maybe_apply_layer_norm(
            x,
            current_layer_norm_application="post",
        )

        return x, masked_attention_probs