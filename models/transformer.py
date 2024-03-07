from .vit_modules import _pair
from .vit_modules import *

class SpatioTempIncModule(nn.Module):
    """Based on the Vision Transformer architecture:

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        input_size: int = 10,
        input_chans: int = 768,
        output_chans: int = 256,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        pre_norm: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Optional[Callable] = None,
        act_layer: Optional[Callable] = None,
        block_fn: Callable = ParallelScalingBlock,
        mlp_layer: Callable = Mlp,
    ):
        """
        Args:
            init_patch_size: Initial patch size.
            n_patch_hw: Number of patches in height and width.
            in_chans: Number of image input channels.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            qk_norm: Enable normalization for qk if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            pre_norm: Enable pre-normalization if True.
            proj_drop_rate: Projection dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            mlp_layer: MLP layer.
        """
        super().__init__()

        self.input_size = input_size
        self.embed_len = self.input_size
        self.embed_dim = embed_dim
        self.input_chans = input_chans
        self.output_chans = output_chans

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_embed = nn.Parameter(torch.randn(1, self.embed_len, self.embed_dim) * 0.02)
        self.norm_pre = norm_layer(self.embed_dim) if pre_norm else nn.Identity()

        if self.embed_dim != self.input_chans:
            self.pre_proj = nn.Sequential(
                nn.Linear(self.input_chans, self.embed_dim),
                nn.GELU(),
            )
        else:
            self.pre_proj = nn.Identity()

        if self.embed_dim != self.output_chans:
            self.post_proj = nn.Linear(self.embed_dim, self.output_chans)
        else:
            self.post_proj = nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, C = x.shape
        x = self.pre_proj(x)
        x = x + self.pos_embed
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.post_proj(x)

        return x