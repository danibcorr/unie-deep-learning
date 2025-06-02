# 3pps
import torch
from torch import nn
from torch.nn import functional as F


class ExpertModel(nn.Module):
    """Modelo experto individual para MoE"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """_summary_

        Args:
            input_dim (int): _description_
            output_dim (int): _description_
            hidden_dim (int): _description_
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            input_tensor (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        return self.model(input_tensor)


class Gating(nn.Module):
    """Gating para seleccionar expertos"""

    def __init__(
        self, input_dim: int, num_experts: int, dropout_rate: float = 0.2
    ) -> None:
        """_summary_

        Args:
            input_dim (int): _description_
            num_experts (int): _description_
            dropout_rate (float, optional): _description_. Defaults to 0.2.
        """

        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate

        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=128),
            nn.Dropout(self.dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=128, out_features=num_experts),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            input_tensor (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        return F.softmax(self.model(input_tensor), dim=-1)


class MoE(nn.Module):
    """Mixture of Experts"""

    def __init__(
        self,
        trained_experts: list[ExpertModel],
        input_dim: int,
        dropout_rate: float = 0.2,
    ) -> None:
        """_summary_

        Args:
            trained_experts (list[nn.Module]): _description_
            input_dim (int): _description_
            dropout_rate (float, optional): _description_. Defaults to 0.2.
        """

        super().__init__()

        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        self.gating_layer = Gating(
            input_dim=self.input_dim,
            num_experts=self.num_experts,
            dropout_rate=self.dropout_rate,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            input_tensor (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        # Obtenemos los pesos del selector
        expert_weights = self.gating_layer(input_tensor)

        # Obtenemos la salida de todos los expertos
        _expert_outputs: list[torch.Tensor] = []
        for expert in self.experts:
            _expert_outputs.append(expert(input_tensor))

        # Stack de salidas [b, output_dim, num_experts]
        expert_outputs = torch.stack(_expert_outputs, dim=-1)

        # [b, num_experts] -> [b, 1, num_experts]
        expert_weights = expert_weights.unsqueeze(1)

        # Suma ponderada de la selección de expertos
        # [b, output_dim, num_experts] * [b, 1, num_experts]
        return torch.sum(expert_outputs * expert_weights, dim=-1)


if __name__ == "__main__":
    input_dim = 10
    output_dim = 5
    num_experts = 3
    batch_size = 32
    hidden_dim = 128

    # Primero crearíamos los expertos, los entrenariamos por separado
    # y luego se introducen en el MoE
    experts = [
        ExpertModel(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        for _ in range(num_experts)
    ]

    # Crear MoE
    moe = MoE(experts, input_dim)

    x = torch.randn(batch_size, input_dim)
    output = moe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {output_dim})")

    # Verificar que los pesos de gating suman 1
    gating_weights = moe.gating_layer(x)
    print(f"Gating weights shape: {gating_weights.shape}")
    print(f"Gating weights sum per sample: {gating_weights.sum(dim=1)}")
