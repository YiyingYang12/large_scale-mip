from jax import numpy as jnp
import flax.linen as nn


class CodebookAttention(nn.Module):
    codebook_dim: int
    depth: int = 1
    num_latents: int = 512
    latent_dim: int = 128
    latent_heads: int = 8
    latent_dim_head: int = 64
    cross_heads: int = 1
    cross_dim_head: int = 64

    def setup(self):
        self.latents = self.param('latents', lambda key, shape: jnp.zeros(shape, dtype=jnp.float32),
                                  (self.num_latents, self.latent_dim))

        self.cross_attend_blocks = [
            PreNorm(self.latent_dim, Attention(self.latent_dim, self.codebook_dim, heads=self.cross_heads,
                                               dim_head=self.cross_dim_head), context_dim=self.codebook_dim),
            PreNorm(self.latent_dim, FeedForward(self.latent_dim))
        ]

        self.self_attend_blocks = []
        for i in range(self.depth):
            self_attn = PreNorm(self.latent_dim, Attention(self.latent_dim, heads=self.latent_heads, dim_head=self.latent_dim_head))
            self_ff = PreNorm(self.latent_dim, FeedForward(self.latent_dim))

            self.self_attend_blocks.append([self_attn, self_ff])

    def __call__(self, codebook):
        """ Useful code items selection.

        Args:
            codebook (jax.numpy.ndarray): [b, n, d]

        Returns:
            x (jax.numpy.ndarray): [b, k, d]
        """
        b = codebook.shape[0]

        x = jnp.tile(self.latents, (b, 1, 1))

        cross_attn, cross_ff = self.cross_attend_blocks

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, context=codebook) + x
        x = cross_ff(x) + x

        # self attention
        for self_attn, self_ff in self.self_attend_blocks:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x
