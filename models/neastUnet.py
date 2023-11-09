import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class NestedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()

        self.inc = DoubleConv(in_channels, init_features)

        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(init_features, init_features * 2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(init_features * 2, init_features * 4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(init_features * 4, init_features * 8))

        self.center = DoubleConv(init_features * 8, init_features * 16)

        self.up3 = nn.Sequential(nn.ConvTranspose3d(init_features * 16, init_features * 8, kernel_size=2, stride=2), DoubleConv(init_features * 16, init_features * 8))
        self.up2 = nn.Sequential(nn.ConvTranspose3d(init_features * 8, init_features * 4, kernel_size=2, stride=2), DoubleConv(init_features * 8, init_features * 4))
        self.up1 = nn.Sequential(nn.ConvTranspose3d(init_features * 4, init_features * 2, kernel_size=2, stride=2), DoubleConv(init_features * 4, init_features * 2))

        self.outc = nn.Conv3d(init_features * 2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        center = self.center(x4)

        x4 = self.up3(torch.cat([center, x4], dim=1))
        x3 = self.up2(torch.cat([x4, x3], dim=1))
        x2 = self.up1(torch.cat([x3, x2], dim=1))

        return self.outc(x2)

if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 1, 128, 128, 128)).to(device)
    model = NestedUNet().to(device)
    outputs = model(inputs)
    print(outputs.shape)

