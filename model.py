from diffusers import UNet2DModel

def create_unet(sample_size=64, in_channels=3):
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=in_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        attention_head_dim=8,
    )
    return model
