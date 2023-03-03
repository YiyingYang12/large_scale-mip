class PreNorm(nn.Module):
    dim: int
    fn: nn.Module
    context_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(name="norm")(x)
        if exists(self.context_dim):
            context = kwargs["context"]
            normed_context = nn.LayerNorm(name="norm_context")(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        x, gates = jnp.split(x, 2, axis=-1)
        return x * nn.gelu(gates)


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
        context_dim = context_dim or query_dim

        q = nn.Dense(inner_dim, use_bias=False)(x)
        context = context or x
        k, v = nn.Dense(inner_dim * 2, use_bias=False)(context).reshape((-1, 2, h, -1)).transpose((0, 2, 1, 3))

        q, k, v = map(lambda t: jnp.reshape(t, (1, -1, h, self.dim_head)), (q, k, v))
        sim = jnp.einsum("bhnd,bhmnd->bhmn", q, k) * self.scale

        if mask is not None:
            mask = jnp.expand_dims(mask, (1, h, 2)).transpose((0, 1, 3, 2))
            max_neg_value = -jnp.finfo(sim.dtype).max
            sim = jnp.where(mask, sim, max_neg_value)

        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("bhmn,bhmnd->bhnd", attn, v)
        out = jnp.reshape(out, (1, -1, inner_dim))
        out = nn.Dense(self.query_dim)(out)

        return out

class CodebookAttention(nn.Module):
    codebook_dim: int
    depth: int = 1
    num_latents: int = 512
    latent_dim: int = 128
    latent_heads: int = 8
    latent_dim_head: int = 64
    cross_heads: int = 1
    cross_dim_head: int = 64

    @nn.compact
    def __call__(self, codebook):
        """ Useful code items selection.

        Args:
            codebook (jax.numpy.ndarray): [b, n, d]

        Returns:
            x (jax.numpy.ndarray): [b, k, d]
        """

        #b = codebook.shape[0]

        x = self.param('latents', lambda key, shape: nn.initializers.normal(key, shape, dtype=jax.numpy.float32), (self.num_latents, self.latent_dim))

        cross_attn = nn.attention.Attention(self.latent_dim, self.codebook_dim, heads=self.cross_heads, key_dim=self.cross_dim_head, value_dim=self.cross_dim_head)
        cross_ff = nn.Dense(self.latent_dim)

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, context=codebook) + x
        x = cross_ff(x) + x

        # self attention
        for _ in range(self.depth):
            self_attn = nn.attention.SelfAttention(self.latent_dim, num_heads=self.latent_heads, head_size=self.latent_dim_head)
            self_ff = nn.Dense(self.latent_dim)

            x = self_attn(x) + x
            x = self_ff(x) + x

        return x




class CoordinateAttention(nn.Module):
    queries_dim: int
    depth: int = 1
    activation: str = "geglu"
    latent_dim: int = 128
    cross_heads: int = 1
    cross_dim_head: int = 64
    decoder_ff: bool = True

    @nn.compact
    def __call__(self, queries, latents):
        cross_attn = Attention(num_heads=self.cross_heads,
                                                qkv_features=self.queries_dim,
                                                attention_axis=1,
                                                projection_dim=self.latent_dim,
                                                kernel_init=nn.initializers.xavier_uniform())
        cross_attn = nn.LayerNorm(name="cross_attn_layer_norm")(cross_attn)
        cross_ffn = nn.Dense(features=self.queries_dim*2 if self.activation=="geglu" else self.queries_dim,
                             kernel_init=nn.initializers.xavier_uniform())
        cross_ffn = nn.Serial(nn.LayerNorm(), cross_ffn, nn.CAE(), name="cross_ff_layer_norm")

        layers = []
        for i in range(self.depth):
            cross_attn_layer = Attention(num_heads=self.cross_heads,
                                                          qkv_features=self.queries_dim,
                                                          attention_axis=1,
                                                          projection_dim=self.latent_dim,
                                                          kernel_init=nn.initializers.xavier_uniform())
            cross_attn_layer = nn.LayerNorm(name=f"cross_attn_{i}_layer_norm")(cross_attn_layer)
            cross_ffn_layer = nn.Dense(features=self.queries_dim*2 if self.activation=="geglu" else self.queries_dim,
                                       kernel_init=nn.initializers.xavier_uniform())
            cross_ffn_layer = nn.Serial(nn.LayerNorm(), cross_ffn_layer, nn.CAE(), name=f"cross_ff_{i}_layer_norm")

            layers.append([cross_attn_layer, cross_ffn_layer])

        x = queries
        for cross_attn, cross_ff in layers:
            x = cross_attn(x) + x
            if cross_ff is not None:
                x = cross_ff(x) + x

        return x
