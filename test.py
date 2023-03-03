def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    dim: int
    fn: nn.Module
    context_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm()(x)
        if exists(self.context_dim):
            context = kwargs["context"]
            context = context.astype(jnp.float32)
            normed_context = nn.LayerNorm()(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        gates = nn.gelu(x2)
        return x1 * gates


class FeedForward(nn.Module):
    dim: int
    mult: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim * self.mult * 2)(x)
        x = GEGLU()(x)
        x = nn.Dense(self.dim)(x)
        return x


class Attention(nn.Module):
    query_dim: int
    context_dim: Optional[int] = None
    heads: int = 8
    dim_head: int = 64

    @nn.compact
    def __call__(self, x, context=None, mask=None):
        h = self.heads
        inner_dim = self.dim_head * h

        q = nn.Dense(inner_dim, use_bias=False)(x)
        context = default(context, x)
        k, v = nn.Dense(inner_dim * 2, use_bias=False)(context).split(2, axis=-1)

        q, k, v = map(lambda t: jnp.reshape(t, (-1, t.shape[1], h, self.dim_head)), (q, k, v))

        sim = jnp.einsum("bhnd, bhkd->bhkn", q, k) * (self.dim_head ** -0.5)

        if exists(mask):
            mask = mask.astype(jnp.bool_)
            max_neg_value = -jnp.finfo(sim.dtype).max
            mask = jnp.reshape(mask, (mask.shape[0], 1, 1, mask.shape[1]))
            mask = lax.broadcast_in_dim(mask, shape=(mask.shape[0], h, sim.shape[2], mask.shape[-1]), broadcast_dimensions=(1,))
            sim = jnp.where(mask, sim, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = jnp.einsum("bhkn, bhvd->bnd", attn, v)
        out = jnp.reshape(out, (-1, out.shape[1], inner_dim))
        out = nn.Dense(self.query_dim)(out)

        return out

 class CodebookAttention(nn.Module):
    def __init__(self, *,
                 codebook_dim,
                 depth: int = 1,
                 num_latents: int = 512,
                 latent_dim: int = 128,
                 latent_heads: int = 8,
                 latent_dim_head: int = 64,
                 cross_heads: int = 1,
                 cross_dim_head: int = 64):

        super().__init__()

        self.latents = nn.Parameter(torch.randn((num_latents, latent_dim), dtype=torch.float32))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, codebook_dim, heads=cross_heads,
                                          dim_head=cross_dim_head), context_dim=codebook_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        self.self_attend_blocks = nn.ModuleList([])
        for i in range(depth):
            self_attn = PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
            self_ff = PreNorm(latent_dim, FeedForward(latent_dim))

            self.self_attend_blocks.append(nn.ModuleList([self_attn, self_ff]))

    def forward(self, codebook):
        """ Useful code items selection.

        Args:
            codebook (torch.Tensor): [b, n, d]

        Returns:
            x (torch.Tensor): [b, k, d]
        """

        #b = codebook.shape[0]

        x = repeat(self.latents, "k d -> b k d", b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, context=codebook) + x
        x = cross_ff(x) + x

        # self attention
        for self_attn, self_ff in self.self_attend_blocks:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x

class CodebookAttention(nn.Module):
    def apply(self, 
              codebook_dim,
              depth: int = 1,
              num_latents: int = 512,
              latent_dim: int = 128,
              latent_heads: int = 8,
              latent_dim_head: int = 64,
              cross_heads: int = 1,
              cross_dim_head: int = 64):
        self.latents = self.param('latents', (num_latents, latent_dim))
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, codebook_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=codebook_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        self.self_attend_blocks = nn.ModuleList([])
        for i in range(depth):
            self_attn = PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head))
            self_ff = PreNorm(latent_dim, FeedForward(latent_dim))
            self.self_attend_blocks.append(nn.ModuleList([self_attn, self_ff]))
        return

    def forward(self, codebook):
        b = codebook.shape[0]
        x = repeat(self.latents, "k d -> b k d", b=b)
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context=codebook) + x
        x = cross_ff(x) + x
        for self_attn, self_ff in self.self_attend_blocks:
            x = self_attn(x) + x
            x = self_ff(x) + x
        return x

class CoordinateAttention(nn.Module):
    def apply(self, 
              queries_dim,
              depth: int = 1,
              activation: str = "geglu",
              latent_dim: int = 128,
              cross_heads: int = 1,
              cross_dim_head: int = 64,
              decoder_ff: bool = True):
        self.cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=latent_dim)
        if activation == "geglu":
            hidden_dim = queries_dim * 2
        else:
            hidden_dim = queries_dim
        self.cross_attend_blocks = nn.ModuleList()
        for i in range(depth):
            cross_attn = PreNorm(queries_dim, Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=latent_dim)
            ffn = nn.Sequential(
                nn.Dense(hidden_dim),
                create_activation(name=activation),
                nn.Dense(queries_dim)
            )
            if i == depth - 1 and decoder_ff:
                cross_ff = PreNorm(queries_dim, ffn)
            else:
                cross_ff = None
            self.cross_attend_blocks.append(nn.ModuleList([cross_attn, cross_ff]))
        return

    def forward(self, queries, latents):
        x = queries
        for cross_attn, cross_ff in self.cross_attend_blocks:
            x = cross_attn(x, context=latents) + x
            if cross_ff is not None:
                x = cross_ff(x) + x
        return x


