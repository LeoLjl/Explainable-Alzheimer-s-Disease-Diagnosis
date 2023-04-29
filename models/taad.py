import torch
import torch.nn as nn
from models.Transformer import TransformerModel
from models.PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
from models.Unet_skipconnection import Unet
from torch.autograd import Variable
import math


class DeepTAAD(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(DeepTAAD, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )


        if self.conv_patch_representation:
            self.conv_x = nn.Conv3d(
                128,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.Unet = Unet(in_channels=1, base_channels=16, num_classes=2)
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(in_features=512, out_features=2)
        self.GAP = nn.AdaptiveAvgPool3d((1, 1, 1))

    def encode(self, x):
        if self.conv_patch_representation:
            
            x1_1, x2_1, x3_1, x = self.Unet(x)  
            x = self.bn(x) 
            x = self.relu(x)
            x = self.conv_x(x) 
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)  

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        x, intmd_x = self.transformer(x)
        
        return x1_1, x2_1, x3_1, x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def cls_branch(self, x):
        x1 = x
        x1 = self._reshape_output(x1) 
        feat = x1 
        N, C, H, W, D = feat.shape  
        x1 = self.GAP(x1)  
        x1 = x1.view(x1.shape[0], -1)  
        y = self.fc(x1)

        for name, parma in self.named_parameters():
            # print(name)
            if name in ['fc.weight']:
                fc_weights = parma
            elif name in ['fc.bias']:
                fc_bias = parma

        fc_weights = fc_weights.view(1, 2, C, 1, 1, 1)  
        fc_weights = Variable(fc_weights, requires_grad=False)

        # attention
        feat = feat.unsqueeze(1) 
        hm = feat * fc_weights  
        hm = hm.sum(2) 
        heatmap = hm
        return y, hm

    def forward(self, x, auxillary_output_layers=[1, 2, 3, 4]):

        x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs = self.encode(x)

        y, heatmap = self.cls_branch(encoder_output)

        decoder_output, out = self.decode(
            x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )
        
        return decoder_output, y, heatmap


    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class TAAD(DeepTAAD):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            num_classes,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(TAAD, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

        self.sigmoid = nn.Sigmoid()

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        # segmentaion part
        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 8)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 8, out_channels=self.embedding_dim // 16)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 16)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 16, out_channels=self.embedding_dim // 32)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 32)

        self.endconv = nn.Conv3d(self.embedding_dim // 32, 1, kernel_size=1)

    def decode(self, x1_1, x2_1, x3_1, x, intmd_x, intmd_layers=[1, 2, 3, 4]):
        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        x8 = encoder_outputs[all_keys[0]] 
        x8 = self._reshape_output(x8)
        out = x8
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8) 

        y4 = self.DeUp4(x8, x3_1) 
        y4 = self.DeBlock4(y4)

        y3 = self.DeUp3(y4, x2_1)  
        y3 = self.DeBlock3(y3)

        y2 = self.DeUp2(y3, x1_1) 
        y2 = self.DeBlock2(y2)
        y = self.endconv(y2) 
        y = self.sigmoid(y)

        return y, out


class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


def get_model(_conv_repr=True, _pe_type="learned"):
    img_dim = 128
    num_classes = 2

    num_channels = 1
    patch_dim = 8
    aux_layers = [1, 2, 3, 4]
    model = TAAD(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


if __name__ == '__main__':
    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((2, 1, 128, 128, 128), device=cuda0)
        _, model = get_model(dataset='brats', _conv_repr=True, _pe_type="learned")
        model.cuda()
        y = model(x)
        print(y)
