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
    num_latents: int
    latent_dim: int
    latent_heads: int
    latent_dim_head: int
    cross_heads: int
    cross_dim_head: int

    @nn.compact
    def __call__(self, x, codebook):
        """ Useful code items selection.

        Args:
            x (jax.interpreters.xla.DeviceArray): [b, k, d]
            codebook (jax.interpreters.xla.DeviceArray): [b, n, d]

        Returns:
            jax.interpreters.xla.DeviceArray: [b, k, d]
        """

        b = codebook.shape[0]

        latents = self.param('latents', nn.initializers.normal(), (self.num_latents, self.latent_dim))
        x = repeat(latents, "k d -> b k d", b=b)

        # cross attention only happens once for Perceiver IO
        x = nn.LayerNormalization()(x)
        x = Attention(self.latent_dim, self.cross_dim_head, self.cross_heads)(x, codebook) + x
        x = nn.LayerNormalization()(x)
        x = nn.Dense(self.latent_dim)(x) + x

        # self attention
        for i in range(self.depth):
            x = nn.LayerNormalization()(x)
            x = Attention(self.latent_dim, self.latent_dim_head, self.latent_heads)(x) + x
            x = nn.LayerNormalization()(x)
            x = nn.Dense(self.latent_dim)(x) + x

        return x


class CoordinateAttention(nn.Module):
    queries_dim: int
    activation: str
    latent_dim: int
    cross_heads: int
    cross_dim_head: int
    decoder_ff: bool

    @nn.compact
    def __call__(self, queries, latents):
        """ Query points features from the latents codebook.

        Args:
            queries (jax.interpreters.xla.DeviceArray): [b, n, c], the sampled points.
            latents (jax.interpreters.xla.DeviceArray): [b, n, k]

        Returns:
            jax.interpreters.xla.DeviceArray: [b, n, c]

        """

        x = queries

        # cross attend from queries to latents
        x = nn.LayerNormalization()(x)
        x = Attention(self.queries_dim, self.latent_dim, self.cross_dim_head, self.cross_heads)(x, latents) + x
        x = nn.LayerNormalization()(x)

        if self.activation == "geglu":
            hidden_dim = self.queries_dim * 2
        else:
            hidden_dim = self.queries_dim

        ffn = nn.Sequential(
            nn.Dense(hidden_dim),
            create_activation(name=self.activation),
            nn.Dense(self.queries_dim)
        )

        if self.decoder_ff:
            x = nn.LayerNormalization()(x)
            x = ffn(x) + x

        return x
