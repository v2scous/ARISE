# src/model.py

import torch
from torch import nn

from .utils import initialize_weights


class DropoutBlock(nn.Module):
    """
    Monte Carlo Dropout block
    - Linear -> Activation -> Dropout
    - activation_fn: "relu", "leaky_relu", "tanh"
    """
    def __init__(self, width, dropout_prob=0.1, activation_fn="relu"):
        super().__init__()
        self.linear = nn.Linear(width, width)

        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "leaky_relu":
            self.activation = nn.LeakyReLU(negative_slope=0.01)
        elif activation_fn == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class ComponentEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        dropout_prob,
        encoder_width,
        encoder_depth,
        mode,
        activation_fn,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, encoder_width),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.component_encoder = nn.ModuleList(
            [
                DropoutBlock(encoder_width, dropout_prob=dropout_prob, activation_fn=activation_fn)
                for _ in range(encoder_depth)
            ]
        )

        self.output_proj = nn.Sequential(
            nn.Linear(encoder_width, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.apply(initialize_weights)
        self.encoder_layer_val = {}
        self.mode = mode

    def forward(self, x):
        # x: (batch_size, comp_num, input_dim)
        X = self.input_proj(x)

        for idx, block in enumerate(self.component_encoder):
            X = block(X)
            self.encoder_layer_val[f"encoder_layer{idx}_values"] = X

        self.embedding = self.output_proj(X)  # (batch_size, comp_num, embedding_dim)

        if self.mode == "Fundamental_props":
            # x[..., -2]가 mole fraction이라고 가정
            self.mask = x[:, :, -2] == 0.0  # (batch_size, comp_num)
            X_embed_masked = self.embedding.masked_fill(self.mask.unsqueeze(-1), 0.0)
            self.X_embed_masked = X_embed_masked
            return self.X_embed_masked
        else:
            return self.embedding


class FunctionDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        output_dim,
        dropout_prob,
        decoder_width,
        decoder_depth,
        activation_fn,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, decoder_width),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self.function_decoder = nn.ModuleList(
            [
                DropoutBlock(decoder_width, dropout_prob=dropout_prob, activation_fn=activation_fn)
                for _ in range(decoder_depth)
            ]
        )

        self.output_proj = nn.Linear(decoder_width, output_dim)

        self.apply(initialize_weights)
        self.decoder_layer_val = {}

    def forward(self, x):
        x = self.input_proj(x)
        for idx, block in enumerate(self.function_decoder):
            x = block(x)
            self.decoder_layer_val[f"decoder_layer{idx}_values"] = x

        output = self.output_proj(x)
        return output


class Pooling(nn.Module):
    """
    Deep Sets 형태 풀링 모듈.
    입력: (batch_size, comp_num, emb_size)
    출력: (batch_size, emb_size)
    """
    def __init__(self, pooling_mode: str = "mean"):
        super().__init__()
        assert pooling_mode in ["mean", "adj_mean"]
        self.mode = pooling_mode

    def forward(self, X_emb, mask=None):
        # X_emb: (batch, comp_num, emb)
        if self.mode == "mean":
            pooled = X_emb.mean(dim=1)

        elif self.mode == "adj_mean":
            pooled = X_emb.sum(dim=1)
            if mask is not None:
                valid_count = (~mask).sum(dim=1)
            else:
                valid_count = X_emb.size(1) * torch.ones(
                    X_emb.size(0), dtype=torch.long, device=X_emb.device
                )
            valid_count = valid_count.clamp(min=1)
            pooled = pooled / valid_count.unsqueeze(-1)

        return pooled


class PermutationLayer(nn.Module):
    def __init__(self, permute: bool = True):
        super().__init__()
        self.permute = permute

    def forward(self, X_emb, force_disable: bool = False):
        """
        X_emb: (batch_size, num_components, embedding_dim)
        force_disable: True이면 permutation 비활성화
        """
        if self.permute and self.training and not force_disable:
            batch_size, num_components, _ = X_emb.shape
            permuted_indices = torch.argsort(
                torch.rand(batch_size, num_components, device=X_emb.device), dim=1
            )
            X_emb = torch.gather(
                X_emb,
                1,
                permuted_indices.unsqueeze(-1).expand(-1, -1, X_emb.shape[2]),
            )

        return X_emb


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_prob, activation_fn):
        super().__init__()
        self.Attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_prob,
            batch_first=True,
        )

    def forward(self, x, mask=None):
        # mask: (batch_size, seq_len) → key_padding_mask: True는 무시됨
        key_padding_mask = mask if mask is not None else None
        attn_output, attn_weights = self.Attention(
            x, x, x, key_padding_mask=key_padding_mask
        )
        self.attn_output = attn_output
        self.attn_weights = attn_weights
        return self.attn_output


class ProposedModel(nn.Module):
    def __init__(
        self,
        mode,
        pooling_mode,
        activation_fn,
        input_dim,
        embedding_dim,
        output_dim,
        encoder_width,
        encoder_depth,
        decoder_width,
        decoder_depth,
        dropout_prob,
        num_heads,
        permute: bool = True,
    ):
        super().__init__()

        self.mode = mode
        self.Component_Encoder = ComponentEncoder(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            dropout_prob=dropout_prob,
            encoder_width=encoder_width,
            encoder_depth=encoder_depth,
            mode=mode,
            activation_fn=activation_fn,
        )
        self.Self_Attention = SelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            activation_fn=activation_fn,
        )
        self.PermutationLayer = PermutationLayer(permute=permute)
        self.Pooling_Layer = Pooling(pooling_mode=pooling_mode)
        # ⚠️ 여기 이름은 "Function_Decoer" 그대로 둬야
        # 기존 state_dict 키와 맞음 (네가 쓴 가중치에 맞추기 위함)
        self.Function_Decoer = FunctionDecoder(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            dropout_prob=dropout_prob,
            decoder_width=decoder_width,
            decoder_depth=decoder_depth,
            activation_fn=activation_fn,
        )

    def forward(self, x, disable_permutation=False):
        embedding = self.Component_Encoder(x)
        mask = self.Component_Encoder.mask  # (batch_size, comp_num)
        embedding = self.PermutationLayer(embedding, force_disable=disable_permutation)
        embedding = self.Self_Attention(embedding, mask)
        embedding = self.Pooling_Layer(embedding, mask)
        output = self.Function_Decoer(embedding)
        return output


def build_viscosity_model_from_config(
    config: dict,
    input_dim: int,
    device: torch.device,
) -> nn.Module:
    """
    JSON config로부터 ProposedModel 생성

    Parameters
    ----------
    config : dict
        JSON에서 읽은 하이퍼파라미터 딕셔너리
    input_dim : int
        len(properties.iloc[:, 0]) + 2
    device : torch.device
        'cpu' 또는 'cuda'
    """

    model = ProposedModel(
        mode=config.get("mode", "Fundamental_props"),
        pooling_mode=config.get("pooling_mode", "adj_mean"),
        activation_fn=config.get("activation_fn", "leaky_relu"),

        input_dim=input_dim,
        embedding_dim=config["embedding_dim"],
        output_dim=config.get("output_dim", 1),

        encoder_width=config["encoder_width"],
        encoder_depth=config["encoder_depth"],
        decoder_width=config["decoder_width"],
        decoder_depth=config["decoder_depth"],
        dropout_prob=config["dropout_prob"],
        num_heads=config["num_heads"],
        permute=config.get("permute", True),
    )

    model.to(device)
    model.eval()
    return model
