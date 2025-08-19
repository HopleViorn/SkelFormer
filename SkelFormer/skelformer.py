import torch
import torch.nn as nn

from hy3dshape.hy3dshape.models.autoencoders.attention_blocks import (
    FourierEmbedder,
    PointCrossAttentionEncoder,
    Transformer,
    CrossAttentionDecoder,
)
from hy3dshape.hy3dshape.models.autoencoders.volume_decoders import VanillaVolumeDecoder
from hy3dshape.hy3dshape.models.autoencoders.surface_extractors import MCSurfaceExtractor
from hy3dshape.hy3dshape.models.autoencoders.model import DiagonalGaussianDistribution


class SkelFormer(nn.Module):
    """
    SkelFormer Network for SDF prediction from a point cloud.

    This network takes a point cloud as input, encodes it into a latent
    representation, and then decodes this representation to predict a
    Signed Distance Function (SDF) field, from which a mesh can be extracted.
    """

    def __init__(
        self,
        *,
        num_latents: int = 256,
        embed_dim: int = 64,
        width: int = 1024,
        heads: int = 16,
        num_decoder_layers: int = 16,
        num_encoder_layers: int = 8,
        pc_size: int = 81920,
        pc_sharpedge_size: int = 0,
        point_feats: int = 4,  # (nx, ny, nz, sharp_edge_label)
        downsample_ratio: int = 20,
        geo_decoder_downsample_ratio: int = 1,
        geo_decoder_mlp_expand_ratio: int = 4,
        num_freqs: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = True,
        **kwargs,
    ):
        """
        Initializes the SkelFormer network.

        Args:
            num_latents (int): Number of latent vectors.
            embed_dim (int): Dimension of the latent embedding space.
            width (int): Width of the transformer layers.
            heads (int): Number of attention heads.
            num_decoder_layers (int): Number of layers in the decoder transformer.
            num_encoder_layers (int): Number of layers in the encoder.
            pc_size (int): Number of points sampled from the surface.
            pc_sharpedge_size (int): Number of points sampled from sharp edges.
            point_feats (int): Number of features per point (excluding xyz).
            downsample_ratio (int): Downsampling ratio in the encoder.
            geo_decoder_downsample_ratio (int): Downsampling ratio for the geo_decoder.
            geo_decoder_mlp_expand_ratio (int): Expansion ratio for MLPs in the geo_decoder.
            num_freqs (int): Number of frequencies for Fourier embedding.
            qkv_bias (bool): Whether to use bias in QKV linear layers.
            qk_norm (bool): Whether to normalize queries and keys.
        """
        super().__init__()

        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs)

        # 1. Encoder: Processes the input point cloud into a set of latent vectors.
        self.encoder = PointCrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            downsample_ratio=downsample_ratio,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            use_ln_post=True,
        )

        # These layers are named to match ShapeVAE for easy weight loading.
        # pre_kl projects to a space for mean and log-variance (2 * embed_dim),
        # but we will only use the mean part.
        self.pre_kl = nn.Linear(width, embed_dim * 2)
        self.post_kl = nn.Linear(embed_dim, width)

        # 2. Transformer: Refines the latent vectors.
        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        # 3. Geometry Decoder: Predicts SDF values for given query points using the refined latents.
        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            mlp_expand_ratio=geo_decoder_mlp_expand_ratio,
            downsample_ratio=geo_decoder_downsample_ratio,
            width=width // geo_decoder_downsample_ratio,
            heads=heads // geo_decoder_downsample_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
        )

        # 4. Utility for generating SDF volume and extracting mesh.
        self.volume_decoder = VanillaVolumeDecoder()
        self.surface_extractor = MCSurfaceExtractor()

    def encode(self, point_cloud: torch.Tensor, sample_posterior: bool = False):
        """
        Encodes a point cloud into a latent representation.

        Args:
            point_cloud (torch.Tensor): Input point cloud with shape (B, N, 3 + F),
                                        where B is batch size, N is number of points,
                                        and F is the number of point features.
            sample_posterior (bool): If True, returns a DiagonalGaussianDistribution object.
                                      If False, returns latent vectors (mean).

        Returns:
            torch.Tensor or DiagonalGaussianDistribution: Latent vectors or distribution.
        """
        pc, feats = point_cloud[:, :, :3], point_cloud[:, :, 3:]
        latents, _ = self.encoder(pc, feats)
        # Get mean and log-variance from the output of pre_kl
        moments = self.pre_kl(latents)
        
        if sample_posterior:
            # Return a DiagonalGaussianDistribution object
            return DiagonalGaussianDistribution(moments, feat_dim=-1)
        else:
            # Return only the mean (for backward compatibility)
            mean, _ = torch.chunk(moments, 2, dim=-1)
            return mean

    def decode_sdf(self, latents: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
        """
        Decodes latents to predict SDF values for given query points.

        Args:
            latents (torch.Tensor): Latent vectors from the encoder.
            query_points (torch.Tensor): 3D coordinates to query for SDF values.

        Returns:
            torch.Tensor: Predicted SDF values.
        """
        # Prepare latents for the geometry decoder
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        
        # Predict SDF values
        sdf_logits = self.geo_decoder(queries=query_points, latents=latents)
        
        return sdf_logits

    def forward(self, point_cloud: torch.Tensor, octree_resolution: int = 256, **kwargs):
        """
        Full reconstruction pipeline: point cloud -> latents -> SDF volume -> mesh.

        Args:
            point_cloud (torch.Tensor): Input point cloud.
            octree_resolution (int): The resolution of the grid for SDF prediction.
            **kwargs: Additional arguments for volume decoding and surface extraction.

        Returns:
            list: A list of reconstructed mesh objects.
        """
        # 1. Encode point cloud to latents
        latents = self.encode(point_cloud)

        # 2. Prepare latents for the geometry decoder
        processed_latents = self.post_kl(latents)
        processed_latents = self.transformer(processed_latents)

        # 3. Generate SDF volume
        # The volume decoder will internally call the geo_decoder for grid points.
        grid_logits = self.volume_decoder(
            latents=processed_latents,
            geo_decoder=self.geo_decoder,
            octree_resolution=octree_resolution,
            **kwargs
        )

        # 4. Extract mesh from SDF volume
        # Note: The original implementation flips the sign of the SDF field.
        mesh_list = self.surface_extractor(-grid_logits, **kwargs)
        
        return mesh_list

    def load_weights_from_shapevae(self, shapevae_state_dict):
        """
        Loads weights from a pretrained ShapeVAE model's state_dict.

        Since SkelFormer's architecture is now aligned with ShapeVAE,
        we can load the weights directly.

        Args:
            shapevae_state_dict (dict): The state dictionary of a ShapeVAE model.
        """
        self.load_state_dict(shapevae_state_dict, strict=True)
        print("Successfully loaded weights from ShapeVAE.")
