import torch

from seagull.nn.modules.glu import GLU
from seagull.nn.modules.linear import Linear
from seagull.nn.modules.module import Module


class FFN(Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        intermediate_dim: int = 2048,
        bias: bool = False,
        activation: str = "swish",
        dropout_proba: float = 0.1,
    ):
        """
        A feed-forward network that applies a GLU layer, followed by a linear transformation.

        Parameters
        ----------
        embedding_dim : int
            The embedding dimension for the GLU and linear layer.
        intermediate_dim : int
            The intermediate dimension for the GLU and linear layer.
        bias : bool
            Whether to include a bias in the GLU and linear layer.
        activation : str
            The type of activation function to use.
        dropout_proba : float
            The dropout probability to use.
        """
        """SwiGLU MLP: https://openreview.net/pdf?id=AL1fq05o7H."""
        super().__init__()

        self.glu = GLU(in_features=embedding_dim, out_features=intermediate_dim, bias=bias, activation=activation)
        self.linear = Linear(
            in_features=intermediate_dim,
            out_features=embedding_dim,
            bias=bias,
            activation=None,
            dropout_proba=dropout_proba,
        )

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass through the network layers using the given input embeddings.

        Parameters
        ----------
        input_embeddings : torch.Tensor
            Input tensor of embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor resulting from forward pass.
        """
        # TODO-5
        return self.linear(self.glu(input_embeddings))