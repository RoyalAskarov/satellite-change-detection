"""One shared encoder, two observation dates, one binary change mask."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

ARCH = 'siamese_resnet18_unet_v1'
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def image_tensor(rgb):
    x = torch.from_numpy(np.array(rgb, dtype=np.float32, copy=True)).permute(2, 0, 1) / 255
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]

class Block(nn.Sequential):
    def __init__(self, inp, out):
        super().__init__(nn.Conv2d(inp, out, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out), nn.ReLU(inplace=True),
                         nn.Conv2d(out, out, 3, padding=1, bias=False),
                         nn.BatchNorm2d(out), nn.ReLU(inplace=True))

class SiameseUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        r = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.layers = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.decoders = nn.ModuleList([Block(512+256,256), Block(256+128,128),
                                       Block(128+64,64), Block(64+64,32)])
        self.head = nn.Sequential(Block(32, 16), nn.Conv2d(16,1,1))

    def encode(self, x):
        x = self.stem(x)
        features = [x]
        x = self.pool(x)
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

    def forward(self, before, after):
        # Joint batch gives both dates the same BatchNorm statistics in training.
        features = self.encode(torch.cat([before, after], dim=0))
        differences = []
        for f in features:
            a, b = f.chunk(2, dim=0)
            differences.append((a-b).abs())
        x = differences[-1]
        for block, skip in zip(self.decoders, reversed(differences[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        x = F.interpolate(x, size=before.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(x)

def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != ARCH:
        raise ValueError('Use levir_cd_best.pt exported by the supplied notebook.')
    if checkpoint.get('mean') != list(MEAN) or checkpoint.get('std') != list(STD):
        raise ValueError('Checkpoint preprocessing does not match this app.')
    model = SiameseUNet()
    model.load_state_dict(checkpoint['model_state'], strict=True)
    return model.to(device).eval(), checkpoint
