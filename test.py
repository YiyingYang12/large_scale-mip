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

    def setup(self):
        self.latents = self.param('latents', nn.initializers.normal(stddev=0.02),
                                  (self.num_latents, self.latent_dim))

        self.cross_attend_blocks = [PreNorm(self.latent_dim, MultiHeadDotProductAttention(self.latent_dim,
                                                                  self.codebook_dim,
                                                                  self.cross_heads,
                                                                  self.cross_dim_head)),
                                    PreNorm(self.latent_dim, Dense(self.latent_dim))]

        self.self_attend_blocks = []
        for i in range(self.depth):
            self_attn = PreNorm(self.latent_dim, MultiHeadDotProductAttention(self.latent_dim,
                                                                self.latent_heads,
                                                                self.latent_dim_head))
            self_ff = PreNorm(self.latent_dim, Dense(self.latent_dim))
            self.self_attend_blocks.append([self_attn, self_ff])

    def forward(self, codebook):
        """ Useful code items selection.

        Args:
            codebook (jax.interpreters.xla.DeviceArray): [b, n, d]

        Returns:
            x (jax.interpreters.xla.DeviceArray): [b, k, d]
        """

        b = codebook.shape[0]

        x = self.latents
        x = x.expand((b,) + x.shape[1:])

        cross_attn, cross_ff = self.cross_attend_blocks

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, context=codebook) + x
        x = cross_ff(x) + x

        # self attention
        for self_attn, self_ff in self.self_attend_blocks:
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
    
    
    
    
class MLP(nn.Module):
  """A PosEnc MLP."""
  codebook: jnp.ndarray
  net_depth: int = 8  # The depth of the first part of MLP.
  net_width: int = 256  # The width of the first part of MLP.
  bottleneck_width: int = 256  # The width of the bottleneck vector.
  net_depth_viewdirs: int = 1  # The depth of the second part of ML.
  net_width_viewdirs: int = 128  # The width of the second part of MLP.
  net_activation: Callable[..., Any] = nn.relu  # The activation function.
  min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
  max_deg_point: int = 12  # Max degree of positional encoding for 3D points.
  weight_init: str = 'he_uniform'  # Initializer for the weights of the MLP.
  skip_layer: int = 4  # Add a skip connection to the output of every N layers.
  skip_layer_dir: int = 4  # Add a skip connection to 2nd MLP every N layers.
  num_rgb_channels: int = 3  # The number of RGB channels.
  deg_view: int = 4  # Degree of encoding for viewdirs or refdirs.
  use_reflections: bool = False  # If True, use refdirs instead of viewdirs.
  use_directional_enc: bool = False  # If True, use IDE to encode directions.
  # If False and if use_directional_enc is True, use zero roughness in IDE.
  enable_pred_roughness: bool = False
  # Roughness activation function.
  roughness_activation: Callable[..., Any] = nn.softplus
  roughness_bias: float = -1.  # Shift added to raw roughness pre-activation.
  use_diffuse_color: bool = False  # If True, predict diffuse & specular colors.
  use_specular_tint: bool = False  # If True, predict tint.
  use_n_dot_v: bool = False  # If True, feed dot(n * viewdir) to 2nd MLP.
  bottleneck_noise: float = 0.0  # Std. deviation of noise added to bottleneck.
  density_activation: Callable[..., Any] = nn.softplus  # Density activation.
  density_bias: float = -1.  # Shift added to raw densities pre-activation.
  density_noise: float = 0.  # Standard deviation of noise added to raw density.
  rgb_premultiplier: float = 1.  # Premultiplier on RGB before activation.
  rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
  rgb_bias: float = 0.  # The shift added to raw colors pre-activation.
  rgb_padding: float = 0.001  # Padding added to the RGB outputs.
  enable_pred_normals: bool = False  # If True compute predicted normals.
  disable_density_normals: bool = False  # If True don't compute normals.
  disable_rgb: bool = False  # If True don't output RGB.
  warp_fn: Callable[..., Any] = None
  basis_shape: str = 'icosahedron'  # `octahedron` or `icosahedron`.
  basis_subdivisions: int = 2  # Tesselation count. 'octahedron' + 1 == eye(3).
  input_dim: int = 3,
  num_latents: int = 8
  latent_dim: int = 1
  latent_heads: int = 4
  latent_dim_head=64
  num_cross_depth: int = 1
  cross_heads: int = 1
  cross_dim_head: int = 64
  decoder_ff: bool = True
  ndepth: int = 1
  activation: str = "softplus"
  
  
  def setup(self,input_dim: int = 3,
               num_latents: int = 8,
               latent_dim: int = 1,
               latent_heads: int = 4,
               latent_dim_head=64,
               num_cross_depth: int = 1,
               cross_heads: int = 1,
               cross_dim_head: int = 64,
               decoder_ff: bool = True,
               ndepth: int = 1,
               activation: str = "softplus",):
    # Make sure that normals are computed if reflection direction is used.
    if self.use_reflections and not (self.enable_pred_normals or
                                     not self.disable_density_normals):
      raise ValueError('Normals must be computed for reflection directions.')

    # Precompute and store (the transpose of) the basis being used.
    self.pos_basis_t = jnp.array(
        geopoly.generate_basis(self.basis_shape, self.basis_subdivisions)).T

    # Precompute and define viewdir or refdir encoding function.
    if self.use_directional_enc:
      self.dir_enc_fn = ref_utils.generate_ide_fn(self.deg_view)
    else:

      def dir_enc_fn(direction, _):
        return coord.pos_enc(
            direction, min_deg=0, max_deg=self.deg_view, append_identity=True)

      self.dir_enc_fn = dir_enc_fn
    codebook_dim = self.codebook.shape[1]
    self.codebook_attn = CodebookAttention(
      codebook_dim=codebook_dim,
      depth=ndepth,
      num_latents=num_latents,
      latent_dim=latent_dim,
      latent_heads=latent_heads,
      latent_dim_head=latent_dim_head,
      cross_heads=cross_heads,
      cross_dim_head=cross_dim_head)
    self.coordinate_attn = CoordinateAttention(
      queries_dim=input_dim,
      depth=num_cross_depth,
      activation=activation,
      latent_dim=latent_dim,
      cross_heads=cross_heads,
      cross_dim_head=cross_dim_head,
      decoder_ff=decoder_ff)


      
  @nn.compact
  def __call__(self,
               rng,
               gaussians,
               viewdirs=None,
               imageplane=None,
               glo_vec=None,
               exposure=None,):
    """Evaluate the MLP.

    Args:
      rng: jnp.ndarray. Random number generator.
      gaussians: a tuple containing:                                           /
        - mean: [..., n, 3], coordinate means, and                             /
        - cov: [..., n, 3{, 3}], coordinate covariance matrices.
      viewdirs: jnp.ndarray(float32), [..., 3], if not None, this variable will
        be part of the input to the second part of the MLP concatenated with the
        output vector of the first part of the MLP. If None, only the first part
        of the MLP will be used with input x. In the original paper, this
        variable is the view direction.
      imageplane: jnp.ndarray(float32), [batch, 2], xy image plane coordinates
        for each ray in the batch. Useful for image plane operations such as a
        learned vignette mapping.
      glo_vec: [..., num_glo_features], The GLO vector for each ray.
      exposure: [..., 1], exposure value (shutter_speed * ISO) for each ray.

    Returns:
      rgb: jnp.ndarray(float32), with a shape of [..., num_rgb_channels].
      density: jnp.ndarray(float32), with a shape of [...].
      normals: jnp.ndarray(float32), with a shape of [..., 3], or None.
      normals_pred: jnp.ndarray(float32), with a shape of [..., 3], or None.
      roughness: jnp.ndarray(float32), with a shape of [..., 1], or None.
    """

    dense_layer = functools.partial(
        nn.Dense, kernel_init=getattr(jax.nn.initializers, self.weight_init)())

    density_key, rng = random_split(rng)
 
 

    def predict_density(means, covs):
      """Helper function to output density."""
      # Encode input position
      if self.warp_fn is not None:
        means, covs = coord.track_linearize(self.warp_fn, means, covs)

      lifted_means, lifted_vars = (
          coord.lift_and_diagonalize(means, covs, self.pos_basis_t))
      x = coord.integrated_pos_enc(lifted_means, lifted_vars,
                                   self.min_deg_point, self.max_deg_point)
      #codebook
      points = x
      b, n, c = x.shape
        #print(b,n,c)
      codebook = self.codebook
      #if codebook.ndim == 2:
      codebook = repeat(codebook, "n d -> b n d", b=b)
      latents = self.codebook_attn(codebook)
      xx = x.view((b, int(n*c/3), 3))
      points = self.coordinate_attn(xx, latents)
      x = points.view((b , n, c))

      inputs = x
      # Evaluate network to produce the output density.
      for i in range(self.net_depth):
        x = dense_layer(self.net_width)(x)
        x = self.net_activation(x)
        if i % self.skip_layer == 0 and i > 0:
          x = jnp.concatenate([x, inputs], axis=-1)
      raw_density = dense_layer(1)(x)[..., 0]  # Hardcoded to a single channel.
      # Add noise to regularize the density predictions if needed.
      if (density_key is not None) and (self.density_noise > 0):
        raw_density += self.density_noise * random.normal(
            density_key, raw_density.shape)
      return raw_density, x

    means, covs = gaussians
    if self.disable_density_normals:
      raw_density, x = predict_density(means, covs)
      raw_grad_density = None
      normals = None
    else:
      # Flatten the input so value_and_grad can be vmap'ed.
      means_flat = means.reshape((-1, means.shape[-1]))
      covs_flat = covs.reshape((-1,) + covs.shape[len(means.shape) - 1:])

      # Evaluate the network and its gradient on the flattened input.
      predict_density_and_grad_fn = jax.vmap(
          jax.value_and_grad(predict_density, has_aux=True), in_axes=(0, 0))
      (raw_density_flat, x_flat), raw_grad_density_flat = (
          predict_density_and_grad_fn(means_flat, covs_flat))

      # Unflatten the output.
      raw_density = raw_density_flat.reshape(means.shape[:-1])
      x = x_flat.reshape(means.shape[:-1] + (x_flat.shape[-1],))
      raw_grad_density = raw_grad_density_flat.reshape(means.shape)

      # Compute normal vectors as negative normalized density gradient.
      # We normalize the gradient of raw (pre-activation) density because
      # it's the same as post-activation density, but is more numerically stable
      # when the activation function has a steep or flat gradient.
      normals = -ref_utils.l2_normalize(raw_grad_density)

    if self.enable_pred_normals:
      grad_pred = dense_layer(3)(x)

      # Normalize negative predicted gradients to get predicted normal vectors.
      normals_pred = -ref_utils.l2_normalize(grad_pred)
      normals_to_use = normals_pred
    else:
      grad_pred = None
      normals_pred = None
      normals_to_use = normals

    # Apply bias and activation to raw density
    density = self.density_activation(raw_density + self.density_bias)

    roughness = None
    if self.disable_rgb:
      rgb = jnp.zeros_like(means)
    else:
      if viewdirs is not None:
        # Predict diffuse color.
        if self.use_diffuse_color:
          raw_rgb_diffuse = dense_layer(self.num_rgb_channels)(x)

        if self.use_specular_tint:
          tint = nn.sigmoid(dense_layer(3)(x))

        if self.enable_pred_roughness:
          raw_roughness = dense_layer(1)(x)
          roughness = (
              self.roughness_activation(raw_roughness + self.roughness_bias))

        # Output of the first part of MLP.
        if self.bottleneck_width > 0:
          bottleneck = dense_layer(self.bottleneck_width)(x)

          # Add bottleneck noise.
          if (rng is not None) and (self.bottleneck_noise > 0):
            key, rng = random_split(rng)
            bottleneck += self.bottleneck_noise * random.normal(
                key, bottleneck.shape)

          x = [bottleneck]
        else:
          x = []

        # Encode view (or reflection) directions.
        if self.use_reflections:
          # Compute reflection directions. Note that we flip viewdirs before
          # reflecting, because they point from the camera to the point,
          # whereas ref_utils.reflect() assumes they point toward the camera.
          # Returned refdirs then point from the point to the environment.
          refdirs = ref_utils.reflect(-viewdirs[..., None, :], normals_to_use)
          # Encode reflection directions.
          dir_enc = self.dir_enc_fn(refdirs, roughness)
        else:
          # Encode view directions.
          dir_enc = self.dir_enc_fn(viewdirs, roughness)

          dir_enc = jnp.broadcast_to(
              dir_enc[..., None, :],
              bottleneck.shape[:-1] + (dir_enc.shape[-1],))

        # Append view (or reflection) direction encoding to bottleneck vector.
        x.append(dir_enc)

        # Append dot product between normal vectors and view directions.
        if self.use_n_dot_v:
          dotprod = jnp.sum(
              normals_to_use * viewdirs[..., None, :], axis=-1, keepdims=True)
          x.append(dotprod)

        # Append GLO vector if used.
        if glo_vec is not None:
          glo_vec = jnp.broadcast_to(glo_vec[..., None, :],
                                     bottleneck.shape[:-1] + glo_vec.shape[-1:])
          x.append(glo_vec)

        # Concatenate bottleneck, directional encoding, and GLO.
        x = jnp.concatenate(x, axis=-1)

        # Output of the second part of MLP.
        inputs = x
        for i in range(self.net_depth_viewdirs):
          x = dense_layer(self.net_width_viewdirs)(x)
          x = self.net_activation(x)
          if i % self.skip_layer_dir == 0 and i > 0:
            x = jnp.concatenate([x, inputs], axis=-1)

      # If using diffuse/specular colors, then `rgb` is treated as linear
      # specular color. Otherwise it's treated as the color itself.
      rgb = self.rgb_activation(self.rgb_premultiplier *
                                dense_layer(self.num_rgb_channels)(x) +
                                self.rgb_bias)

      if self.use_diffuse_color:
        # Initialize linear diffuse color around 0.25, so that the combined
        # linear color is initialized around 0.5.
        diffuse_linear = nn.sigmoid(raw_rgb_diffuse - jnp.log(3.0))
        if self.use_specular_tint:
          specular_linear = tint * rgb
        else:
          specular_linear = 0.5 * rgb

        # Combine specular and diffuse components and tone map to sRGB.
        rgb = jnp.clip(
            image.linear_to_srgb(specular_linear + diffuse_linear), 0.0, 1.0)

      # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
      rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

    return dict(
        density=density,
        rgb=rgb,
        raw_grad_density=raw_grad_density,
        grad_pred=grad_pred,
        normals=normals,
        normals_pred=normals_pred,
        roughness=roughness,
    )
