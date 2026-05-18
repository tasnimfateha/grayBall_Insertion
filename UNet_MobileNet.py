import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """
    Two convolution layers used inside U-Net.
    Each convolution is followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block of U-Net.

    It reduces spatial size by 2 using MaxPool,
    then applies two convolution layers.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    Upsampling block of U-Net.

    It upsamples the previous feature map,
    concatenates it with a skip connection from the encoder,
    then applies two convolution layers.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = self.up(x)

        # Pad if shapes differ by 1 pixel due to rounding
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)

        x = F.pad(
            x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )

        x = torch.cat([skip, x], dim=1)

        return self.conv(x)


class MobileNetV3GlobalEncoder(nn.Module):
    """
    Extracts global scene features using MobileNetV3.

    Input:
        full scene image [B, 3, 224, 224]

    Output:
        spatial feature map [B, 256, 32, 32]
    """

    def __init__(self, output_channels=256, freeze_backbone=True, unfreeze_last_blocks=3):
        super().__init__()

        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        mobilenet = models.mobilenet_v3_large(weights=weights)

        self.features = mobilenet.features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

            # Fine-tune only the last few MobileNet blocks
            if unfreeze_last_blocks > 0:
                for layer in list(self.features.children())[-unfreeze_last_blocks:]:
                    for param in layer.parameters():
                        param.requires_grad = True

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Linear(
            960,
            output_channels * 8 * 8,
        )

        self.output_channels = output_channels

    def forward(self, full_scene):
        batch_size = full_scene.size(0)

        features = self.features(full_scene)          # [B, 960, 7, 7]
        features = self.global_pool(features)         # [B, 960, 1, 1]
        features = features.flatten(start_dim=1)      # [B, 960]

        features = self.projection(features)          # [B, 256*8*8]
        features = features.view(
            batch_size,
            self.output_channels,
            8,
            8,
        )                                            # [B, 256, 8, 8]

        features = F.interpolate(
            features,
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )                                            # [B, 256, 32, 32]

        return features


class UNetMobileNetV3(nn.Module):
    """
    U-Net + MobileNetV3 model for grey ball insertion.

    Inputs:
        local_crop:
            [B, 3, 256, 256]
            Aligned crop around the target ball position.

        global_scene:
            [B, 3, 224, 224]
            Full aligned scene used for global illumination context.

    Output:
        predicted_ball:
            [B, 3, 256, 256]
            RGB prediction of the grey ball patch.
    """

    def __init__(self, input_channels=3, output_channels=3):
        super().__init__()

        # Local U-Net encoder
        self.enc1 = DoubleConv(input_channels, 32)   # 256x256
        self.enc2 = DownBlock(32, 64)                # 128x128
        self.enc3 = DownBlock(64, 128)               # 64x64
        self.enc4 = DownBlock(128, 256)              # 32x32

        # Global MobileNetV3 encoder
        self.global_encoder = MobileNetV3GlobalEncoder(
            output_channels=256,
            freeze_backbone=True,
            unfreeze_last_blocks=3,
        )

        # Fusion of local and global features
        self.bottleneck = DoubleConv(512, 256)

        # U-Net decoder
        self.dec1 = UpBlock(256, 128, 128)           # 64x64
        self.dec2 = UpBlock(128, 64, 64)             # 128x128
        self.dec3 = UpBlock(64, 32, 32)              # 256x256

        self.output_layer = nn.Conv2d(
            32,
            output_channels,
            kernel_size=1,
        )

        self.activation = nn.Sigmoid()

    def forward(self, local_crop, global_scene):
        # -------------------------
        # Local U-Net encoder
        # -------------------------
        skip1 = self.enc1(local_crop)     # [B, 32, 256, 256]
        skip2 = self.enc2(skip1)          # [B, 64, 128, 128]
        skip3 = self.enc3(skip2)          # [B, 128, 64, 64]
        local_features = self.enc4(skip3) # [B, 256, 32, 32]

        # -------------------------
        # Global scene features
        # -------------------------
        global_features = self.global_encoder(global_scene)  # [B, 256, 32, 32]

        # -------------------------
        # Fuse local and global features
        # -------------------------
        fused = torch.cat(
            [local_features, global_features],
            dim=1,
        )  # [B, 512, 32, 32]

        bottleneck = self.bottleneck(fused)  # [B, 256, 32, 32]

        # -------------------------
        # U-Net decoder
        # -------------------------
        x = self.dec1(bottleneck, skip3)  # [B, 128, 64, 64]
        x = self.dec2(x, skip2)           # [B, 64, 128, 128]
        x = self.dec3(x, skip1)           # [B, 32, 256, 256]

        output = self.output_layer(x)     # [B, 3, 256, 256]
        output = self.activation(output)  # values between 0 and 1

        return output


if __name__ == "__main__":
    model = UNetMobileNetV3()

    local_crop = torch.randn(2, 3, 256, 256)
    global_scene = torch.randn(2, 3, 224, 224)

    output = model(local_crop, global_scene)

    print("Local crop shape:", local_crop.shape)
    print("Global scene shape:", global_scene.shape)
    print("Output shape:", output.shape)