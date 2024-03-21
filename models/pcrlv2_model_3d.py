import torch
import torch.nn as nn
import torch.nn.functional as F


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act, norm):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)

        if norm == 'bn':
            self.bn1 = nn.BatchNorm3d(num_features=out_chan, momentum=0.1, affine=True)
        elif norm == 'gn':
            self.bn1 = nn.GroupNorm(num_groups=8, num_channels=out_chan, eps=1e-05, affine=True)
        elif norm == 'in':
            self.bn1 = nn.InstanceNorm3d(num_features=out_chan, momentum=0.1, affine=True)
        else:
            raise ValueError('normalization type {} is not supported'.format(norm))

        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif act == 'sigmoid':
            self.activation = nn.Sigmoid()

        else:
            raise ValueError('activation type {} is not supported'.format(act))

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, norm, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act, norm)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act, norm)
    else:
        layer1 = LUConv(in_channel, 32 * (2 ** depth), act, norm)
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, act, norm)

    return nn.Sequential(layer1, layer2)


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act, norm, skip_conn=False):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        if skip_conn:
            self.ops = _make_nConv(inChans + outChans//2, depth, act, norm, double_chnnel=True)
        else:
            self.ops = _make_nConv(outChans, depth, act, norm, double_chnnel=True)
        channels = 32 * (2 ** depth) * 2
        self.bn = nn.BatchNorm1d(channels)
        self.predictor_head = nn.Sequential(nn.Linear(channels, 2 * channels),
                                            nn.BatchNorm1d(2 * channels),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2 * channels, channels))
        self.deep_supervision_head = LUConv(channels, 1, 'sigmoid', norm)

    def forward(self, x, skip_x = None, pretrain=True):
        b = x.shape[0]
        out_up_conv = self.up_conv(x)
        if skip_x is not None:
            out = torch.cat((out_up_conv, skip_x), 1)
        else:
            out = out_up_conv
        x = self.ops(out)
        if pretrain:
            x_pro = F.adaptive_avg_pool3d(x, (1, 1, 1))
            x_pro = x_pro.view(b, -1)
            x_pro = self.bn(x_pro)
            x_pre = self.predictor_head(x_pro)
            x_mask = self.deep_supervision_head(x)
            return x, x_pro, x_pre, x_mask
        return x


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act, norm):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act, norm)

    def forward(self, x):
        return self.ops(x)


class PCRLv23d(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', norm='bn', in_channels=1, low_dim=128, student=False, skip_conn=False):
        super(PCRLv23d, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.down_tr64 = DownTransition(in_channels, 0, act, norm)
        self.down_tr128 = DownTransition(64, 1, act, norm)
        self.down_tr256 = DownTransition(128, 2, act, norm)
        self.down_tr512 = DownTransition(256, 3, act, norm)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.up_tr256 = UpTransition(512, 512, 2, act, norm, skip_conn=skip_conn)
        self.up_tr128 = UpTransition(256, 256, 1, act, norm, skip_conn=skip_conn)
        self.up_tr64 = UpTransition(128, 128, 0, act, norm, skip_conn=skip_conn)
        self.out_tr = OutputTransition(64, n_class)
        self.sigmoid = nn.Sigmoid()
        self.skip_conn=skip_conn

    def forward(self, x, local=False):
        b = x.shape[0]
        self.skip_out64 = self.down_tr64(x)
        self.skip_out128 = self.down_tr128(self.maxpool(self.skip_out64))
        self.skip_out256 = self.down_tr256(self.maxpool(self.skip_out128))
        self.out512 = self.down_tr512(self.maxpool(self.skip_out256))
        middle_masks = []
        middle_features = []
        if self.skip_conn:
            out_up_256, pro_256, pre_256, middle_masks_256 = self.up_tr256(self.out512, skip_x=self.skip_out256)
        else:
            out_up_256, pro_256, pre_256, middle_masks_256 = self.up_tr256(self.out512)
        if self.skip_conn:
            out_up_128, pro_128, pre_128, middle_masks_128 = self.up_tr128(out_up_256, skip_x=self.skip_out128)
        else:
            out_up_128, pro_128, pre_128, middle_masks_128 = self.up_tr128(out_up_256)
        if self.skip_conn:
            out_up_64, pro_64, pre_64, middle_masks_64 = self.up_tr64(out_up_128, skip_x=self.skip_out64)
        else:
            out_up_64, pro_64, pre_64, middle_masks_64 = self.up_tr64(out_up_128)

        if not local:
            middle_masks.append(F.interpolate(middle_masks_256, scale_factor=4, mode='trilinear'))
            middle_masks.append(F.interpolate(middle_masks_128, scale_factor=2, mode='trilinear'))
            middle_masks.append(middle_masks_64)
        middle_features.append([pro_256, pre_256])
        middle_features.append([pro_128, pre_128])
        middle_features.append([pro_64, pre_64])
        # normal decoder
        out = self.out_tr(out_up_64)

        return out, middle_features, middle_masks


class Cluster3d(nn.Module):
    def __init__(self, n_clusters=10, act='relu', norm='bn', in_channels=1):
        super(Cluster3d, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.down_tr64 = DownTransition(in_channels, 0, act, norm)
        self.down_tr128 = DownTransition(64, 1, act, norm)
        self.down_tr256 = DownTransition(128, 2, act, norm)
        self.down_tr512 = DownTransition(256, 3, act, norm)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.sigmoid = nn.Sigmoid()

        # Clustering Pretask Head
        self.patch_dim = (8,8,4)  # P1 x P2 x P3
        self.patch_num = 512  # N  # Does nothing, just useful info
        self.emb_dim = 64  # D
        self.proto_num = n_clusters  # K
        self.cluster_projection_head = nn.Linear(self.patch_dim[0]*self.patch_dim[1]*self.patch_dim[2], self.emb_dim)  # Projection head
        self.prototypes = nn.Linear(self.emb_dim, self.proto_num, bias=False)

    def forward(self, x, local=False):
        b = x.shape[0]
        self.skip_out64 = self.down_tr64(x)
        self.skip_out128 = self.down_tr128(self.maxpool(self.skip_out64))
        self.skip_out256 = self.down_tr256(self.maxpool(self.skip_out128))
        self.out512 = self.down_tr512(self.maxpool(self.skip_out256))

        # Flatten spatial dims of feature map
        B, N, PH, PW, PD = self.out512.shape  # Batch, Number of patches, Patch spatial dims
        encoder_output = self.out512.reshape(B, N, PH*PW*PD)

        # Get embeddings and output preds
        emb = self.cluster_projection_head(encoder_output)
        out = self.prototypes(emb)

        return emb, out


class TraceWrapper(torch.nn.Module):
    # Wrapper class for PCRLv23D for tracing the model with tensorboard. It's because its forward outputs a tuple and writer.add_graph wants a tensor output
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        out = self.model(input)
        return out[0]


class SegmentationModel(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', norm='bn', in_channels=1, low_dim=128, student=False, skip_conn=False):
        super(SegmentationModel, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.down_tr64 = DownTransition(in_channels, 0, act, norm)
        self.down_tr128 = DownTransition(64, 1, act, norm)
        self.down_tr256 = DownTransition(128, 2, act, norm)
        self.down_tr512 = DownTransition(256, 3, act, norm)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.up_tr256 = UpTransition(512, 512, 2, act, norm, skip_conn=skip_conn)
        self.up_tr128 = UpTransition(256, 256, 1, act, norm, skip_conn=skip_conn)
        self.up_tr64 = UpTransition(128, 128, 0, act, norm, skip_conn=skip_conn)
        self.out_tr = OutputTransition(64, n_class)
        self.sigmoid = nn.Sigmoid()
        self.skip_conn = skip_conn

    def forward(self, x):
        b = x.shape[0]
        self.skip_out64 = self.down_tr64(x)
        self.skip_out128 = self.down_tr128(self.maxpool(self.skip_out64))
        self.skip_out256 = self.down_tr256(self.maxpool(self.skip_out128))
        self.out512 = self.down_tr512(self.maxpool(self.skip_out256))
        middle_masks = []
        middle_features = []
        if self.skip_conn:
            out_up_256 = self.up_tr256(self.out512, skip_x=self.skip_out256, pretrain=False)
        else:
            out_up_256 = self.up_tr256(self.out512, pretrain=False)
        if self.skip_conn:
            out_up_128 = self.up_tr128(out_up_256, skip_x=self.skip_out128, pretrain=False)
        else:
            out_up_128 = self.up_tr128(out_up_256, pretrain=False)
        if self.skip_conn:
            out_up_64 = self.up_tr64(out_up_128, skip_x=self.skip_out64, pretrain=False)
        else:
            out_up_64 = self.up_tr64(out_up_128, pretrain=False)

        # normal decoder
        out = self.out_tr(out_up_64, )
        return out
