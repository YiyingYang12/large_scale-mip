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
        b, n, d = codebook.shape

        latents = self.param('latents', lambda key: jax.random.normal(key, (self.num_latents, self.latent_dim)))
        x = jnp.tile(latents, (b, 1, 1))

        # cross attention
        cross_attn = nn.Sequential(
            nn.LayerNorm(),
            nn.MultiHeadAttention(num_heads=self.cross_heads, key_dim=self.latent_dim, value_dim=self.codebook_dim,
                                  dropout=0.1),
        )

        cross_ff = nn.Sequential(
            nn.LayerNorm(),
            nn.Dense(features=self.latent_dim),
            nn.Dropout(rate=0.1),
            nn.Dense(features=self.latent_dim),
        )

        x = x + cross_attn(x, key=codebook, value=codebook)
        x = x + cross_ff(x)

        # self attention
        for _ in range(self.depth):
            self_attn = nn.Sequential(
                nn.LayerNorm(),
                nn.MultiHeadAttention(num_heads=self.latent_heads, key_dim=self.latent_dim, value_dim=self.latent_dim_head,
                                      dropout=0.1),
            )

            self_ff = nn.Sequential(
                nn.LayerNorm(),
                nn.Dense(features=self.latent_dim),
                nn.Dropout(rate=0.1),
                nn.Dense(features=self.latent_dim),
            )

            x = x + self_attn(x, key=x, value=x)
            x = x + self_ff(x)

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
        """ Query points features from the latents codebook.

        Args:
            queries (jax.numpy.ndarray): [b, n, c], the sampled points.
            latents (jax.numpy.ndarray): [b, n, k]

        Returns:
            x (jax.numpy.ndarray): [b, n, c]

        """

        x = queries

        # cross attend from queries to latents
        for i in range(self.depth):
            cross_attn = PreNorm(self.queries_dim, Attention(self.queries_dim, self.latent_dim,
                                            heads=self.cross_heads, dim_head=self.cross_dim_head), 
                                            context_dim=self.latent_dim)

            ffn = nn.Sequential(
                nn.Dense(self.queries_dim * 2, use_bias=True),
                GEGLU() if self.activation == "geglu" else nn.gelu,
                nn.Dense(self.queries_dim, use_bias=True)
            )

            if i == self.depth - 1 and self.decoder_ff:
                cross_ff = PreNorm(self.queries_dim, ffn)
            else:
                cross_ff = None

            cross_attn_output = cross_attn(x, context=latents)
            x = cross_attn_output + x

            if cross_ff is not None:
                x = cross_ff(x) + x

        return x

    
    
