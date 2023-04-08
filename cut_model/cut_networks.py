import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def print_layer(layer, x):

    if hasattr(layer, '__iter__'):
        for l in layer:
            x = l(x)
            print(l.__class__.__name__)
            print(x.size())
            print(f"min: {x.min():.3f} max: {x.max():.3f} mean: {x.mean():.3f}" )
    else:
        x = layer(x)
        print(layer.__class__.__name__)
        print(x.size())
        print(f"min: {x.min():.3f} max: {x.max():.3f} mean: {x.mean():.3f}" )

    return x

def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight, gain=0.02)
        nn.init.constant_(layer.bias, val=0.0)

class ResNetBlock(nn.Module):
    def __init__(self, channels=256):
        super(ResNetBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
        )

        self.model.apply(init_weights)

    def forward(self, x):
        return x + self.model(x)

class CUT_G_encoder(nn.Module):
    def __init__(self):
        super(CUT_G_encoder, self).__init__()

        self.layer0 = nn.ReflectionPad2d(3)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            #Downsample(128),
            nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            #Downsample(256),
        )

        self.layer4 = ResNetBlock(256)
        self.layer5 = nn.Sequential(
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
        )
        self.layer6 = nn.Sequential(
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
            ResNetBlock(256),
        )

        self.layers = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            ]
        
        for l in self.layers:
            l.apply(init_weights)

    def forward(self, x):
        feats = []
        feat = x

        for l in self.layers:
            #x = print_layer(l, x)
            feat = l(feat)
            feats.append(feat)

        return feats, feats[-1] #intermediate features and encoder output
    
class CUT_G_decoder(nn.Module):
    def __init__(self):
        super(CUT_G_decoder, self).__init__()

        self.model = nn.Sequential(
            #Upsample(256,1),
            #nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            #Upsample(128,1),
            #nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

        self.model.apply(init_weights)

    def forward(self, x):
        #x = print_layer(self.model, x)
        #return x
        return self.model(x)

class CUT_D(nn.Module):
    def __init__(self):
        super(CUT_D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2, True),

            #Downsample(64, 1),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),

            #Downsample(128, 1),
            nn.Conv2d(128, 256, kernel_size=4, padding=1, stride=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),

            #Downsample(256, 1),
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)

class CUT_H(nn.Module):
    def __init__(self, output_channels=256, input_channels=[3,128,256,256,256,256,256]):
        super(CUT_H, self).__init__()

        self.models = []

        for ic in input_channels:
            self.models.append(nn.Sequential(
                nn.Linear(ic, output_channels),
                nn.ReLU(True),
                nn.Linear(output_channels, output_channels)
            ))

        self.models = nn.ModuleList(self.models)

        self.models.apply(init_weights)

    def forward(self, feats, patch_ids):

        latent_feats = []

        for i in range(len(patch_ids)):
            feat = feats[i]
            #B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            patch_id = patch_ids[i]

            x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)
            x_sample = self.models[i](x_sample)

            x_sample = F.normalize(x_sample, p=2, dim=1)
            latent_feats.append(x_sample)

        return latent_feats
    
class PatchNCELoss(nn.Module):
    def __init__(self, n_patches, tempurature, device):
        super(PatchNCELoss, self).__init__()

        self.t = tempurature
        self.target = torch.arange(start=0, end=n_patches, dtype=torch.long, device=device).unsqueeze(0)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, q, k):
        dim = q.shape[1]
        k = k.detach()

        q = q.view(1, -1, dim)
        k = k.view(1, -1, dim)
        v = torch.bmm(q, k.transpose(2, 1)) / self.t

        #diagonal is similarity with q and k patches with same patch_id
        loss = self.cross_entropy_loss(v, self.target)
        
        return loss
    
def get_filter(filt_size=3):
    if(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

class Downsample(nn.Module):
    def __init__(self, channels, pad=0):
        super(Downsample, self).__init__()

        filt = get_filter(3)
        filt = filt[None, None, :, :].repeat((channels, 1, 1, 1))
        self.register_buffer('filter', filt)
        self.pad = nn.ReflectionPad2d(pad)

    def forward(self, x):
        x = self.pad(x)
        #return F.interpolate(x, scale_factor=0.5, mode='bilinear', antialias=True)
        
        return F.conv2d(x, self.filter, stride=2, padding=1, groups=x.shape[1])

class Upsample(nn.Module):
    def __init__(self, channels, pad=0):
        super(Upsample, self).__init__()

        filt = get_filter(4)
        filt = filt[None, None, :, :].repeat((channels, 1, 1, 1)) * (2**2) # (stride**2)
        self.register_buffer('filter', filt)
        self.pad = nn.ReplicationPad2d(pad)

    def forward(self, x):
        x = self.pad(x)
        #return F.interpolate(x, scale_factor=2.0, mode='bilinear', antialias=True)
        x = F.conv_transpose2d(x, self.filter, stride=2, padding=2, groups=x.shape[1])
        return x[:,:,1:-1,1:-1]
        
