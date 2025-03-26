import geoopt
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
import manifolds
from .wrapped_normal import WrappedNormal
from .riemannian_normal import RiemannianNormal
from torch.distributions import Normal

from .vae import VAE
from .utils import Constants

from .manifold_layers import GyroplaneConvLayer


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()

        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=True))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        elif mode_upsampling == 2:
            ops.append(nn.Upsample(scale_factor=stride, mode="nearest"))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, manifold, hidden_dim=300, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        self.manifold = manifold
        self.hidden_dim = hidden_dim

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.block_gap = convBlock(3, n_filters * 16, 1, normalization=normalization)

        self.fc21 = nn.Linear(1, manifold.coord_dim)
        self.fc22 = nn.Linear(1, manifold.coord_dim)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        b = x5.shape[0]
        gap_x = self.block_gap(x5)
        gap_x = gap_x.view(1, -1)
        s = int(gap_x.shape[1] / b)

        mu = self.fc21(gap_x.T).view(b, s, 2)
        mu_sp = F.softplus(self.fc22(gap_x.T)).view(b, s, 2) + Constants.eta

        res = [x1, x2, x3, x4, x5]

        return res, (mu, mu_sp, self.manifold)


class Encoder_H(nn.Module):
    def __init__(self, manifold, hidden_dim=300, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder_H, self).__init__()
        self.has_dropout = has_dropout
        self.manifold = manifold
        self.hidden_dim = hidden_dim

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)

        self.block_gap = convBlock(3, n_filters * 16, 1, normalization=normalization)

        self.fc21 = nn.Linear(1, manifold.coord_dim)
        self.fc22 = nn.Linear(1, manifold.coord_dim)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

        #self.hyp = geoopt.manifolds.PoincareBall(c=1.0, learnable=True)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        if self.has_dropout:
            x5 = self.dropout(x5)

        b = x5.shape[0]
        gap_x = self.block_gap(x5)

        c_gap = gap_x.clone()

        c_gap = c_gap.view(1, -1)
        s = int(c_gap.shape[1] / b)

        mu = self.fc21(c_gap.T)
        mu = self.manifold.expmap0(mu).view(b, s, 2)
        #mu = self.hyp.expmap0(mu).view(b, s, 2)

        fc_out = self.fc22(c_gap.T)
        mu_sp = F.softplus(fc_out).view(b, s, 2) + Constants.eta

        res = [x1, x2, x3, x4, x5]

        return res, (mu, mu_sp, self.manifold)


class Decoder(nn.Module):
    def __init__(self, manifold, hidden_dim, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout
        self.manifold = manifold

        self.lin = nn.Linear(manifold.coord_dim, hidden_dim)  # hidden_dim
        self.act = nn.ReLU()


        # self.dim_reduction = nn.ConvTranspose3d(in_channels=hidden_dim, out_channels=n_filters * 16,
        #                                             kernel_size=(5, 5, 5), padding=(0, 0, 0))  # use Relu after
        self.dim_reduction = nn.ConvTranspose3d(in_channels=hidden_dim, out_channels=n_filters * 16,
                                                kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # use Relu after

        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features, zs):
        batch_size = zs.shape[0]
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        mu_lin = self.lin(zs)
        v = mu_lin.view(batch_size, -1,  6, 6, 6)
        dim_red = self.act(self.dim_reduction(v))

        x5_up = self.block_five_up(dim_red)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        out_seg = self.out_conv(x9)

        return out_seg, dim_red


class Decoder_H(nn.Module):
    def __init__(self, manifold, hidden_dim=100, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0, dataset_name="LA"):
        super(Decoder_H, self).__init__()
        self.has_dropout = has_dropout
        self.manifold = manifold
        self.dataset_name = dataset_name

        convBlock = ConvBlock if not has_residual else ResidualConvBlock
        self.gyro_conv = GyroplaneConvLayer(in_features=manifold.coord_dim, out_channels=hidden_dim, kernel_size=1,
                                            manifold=manifold)  # use Relu after

        # self.dim_reduction = nn.ConvTranspose3d(in_channels=hidden_dim, out_channels=n_filters * 16,
        #                                         kernel_size=(6, 6, 4), padding=(0, 0, 0))  # use Relu after

        self.dim_reduction = nn.ConvTranspose3d(in_channels=hidden_dim, out_channels=n_filters * 16,
                                                kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))  # use Relu after

        self.act = nn.ReLU()

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features, zs):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        gyro_out = self.act(self.gyro_conv(zs))

        dim_red = self.act(self.dim_reduction(gyro_out))

        x5_up = self.block_five_up(dim_red)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        out_seg = self.out_conv(x9)

        return out_seg, dim_red


class VNet(VAE):
    def __init__(self, params, hidden_dim=256, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold_euc)(params.latent_dim, c)
        dataset_name = params.dataset_name
        super(VNet, self).__init__(
            eval(params.prior_euc),  # prior distribution
            eval(params.posterior_euc),  # posterior distribution
            dist.Normal,  # likelihood distribution
            eval('Encoder')(manifold, hidden_dim, n_channels, n_classes, n_filters, normalization, has_dropout,
                              has_residual),
            eval('Decoder')(manifold, hidden_dim, n_channels, n_classes, n_filters, normalization, has_dropout,
                              has_residual, 0),
            params
        )
        self.manifold = manifold
        self.c = c
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)

    def forward(self, input):
        features, enc_mu = self.enc(input)
        mu, mu_sp, _ = enc_mu
        qz_x = self.qz_x(*enc_mu)
        zs = qz_x.rsample(torch.Size([1]))
        zs_H = self.manifold.expmap0(zs)

        zs = zs.squeeze(0)
        out_seg1, feat = self.dec(features, zs)

        zs = zs.permute(1,0,2)

        zs_H = zs_H.squeeze(0)
        zs_H = zs_H.permute(1, 0, 2)

        return out_seg1, feat, enc_mu, zs, zs_H


class VNet_H(VAE):
    def __init__(self, params, hidden_dim=256, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        c = nn.Parameter(params.c * torch.ones(1), requires_grad=False)
        manifold = getattr(manifolds, params.manifold)(params.latent_dim, c)
        dataset_name = params.dataset_name
        super(VNet_H, self).__init__(
            eval(params.prior),  # prior distribution
            eval(params.posterior),  # posterior distribution
            dist.Normal,        # likelihood distribution
            eval('Encoder_H')(manifold, hidden_dim, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual),
            eval('Decoder_H')(manifold, hidden_dim, n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0),
            params
        )

        self.manifold = manifold
        self.c = c
        self._pz_mu = nn.Parameter(torch.zeros(1, params.latent_dim), requires_grad=False)
        self._pz_logvar = nn.Parameter(torch.zeros(1, 1), requires_grad=params.learn_prior_std)

    def forward(self, input):
        features, enc_mu = self.enc(input)
        mu, mu_sp, _ = enc_mu
        qz_x = self.qz_x(*enc_mu)

        zs = qz_x.rsample(torch.Size([1]))
        zs_E = self.manifold.logmap0(zs)

        zs = zs.squeeze(0)
        zs = zs.permute(1,0,2)

        zs_E = zs_E.squeeze(0)
        zs_E = zs_E.permute(1, 0, 2)

        out_seg1, feat = self.dec(features, zs)

        return out_seg1, feat, enc_mu, zs, zs_E


if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info

    model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)

    macs, params = get_model_complexity_info(model, (1, 96, 96, 96), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    import ipdb;

    ipdb.set_trace()