import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.layers import (
    ResnetBlockFC, CResnetBlockConv1d,
    CBatchNorm1d, CBatchNorm1d_legacy,
)


def maxpool(x, dim=-1, keepdim=False):
    ''' Performs a maxpooling operation.

    Args:
        x (tensor): input
        dim (int): dimension of pooling
        keepdim (bool): whether to keep dimensions
    '''
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        # self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=512, is_pool=True):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.is_pool = is_pool
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        if self.is_pool:
            net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class TemporalResnetPointnet(nn.Module):
    ''' Temporal PointNet-based encoder network.

    The input point clouds are concatenated along the hidden dimension,
    e.g. for a sequence of length L, the dimension becomes 3xL = 51.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        use_only_first_pcl (bool): whether to use only the first point cloud
    '''

    def __init__(self, c_dim=3, dim=51, hidden_dim=128,
                 use_only_first_pcl=False, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.use_only_first_pcl = use_only_first_pcl

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        batch_size, n_steps, n_pts, _ = x.shape

        if len(x.shape) == 4 and self.use_only_first_pcl:
            x = x[:, 0]
        elif len(x.shape) == 4:
            x = x.transpose(1, 2).contiguous().view(batch_size, n_pts, -1)

        net = self.fc_pos(x)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # aux = aux_code.unsqueeze(1).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # aux = aux_code.unsqueeze(1).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # aux = aux_code.unsqueeze(1).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        # aux = aux_code.unsqueeze(1).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)
        net = self.pool(net, dim=1)
        c = self.fc_c(self.actvn(net))

        return c


class Cond_ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=3, dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p, offset_code):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        offset = offset_code.unsqueeze(1).expand(net.size())
        net = torch.cat([net, pooled, offset], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        offset = offset_code.unsqueeze(1).expand(net.size())
        net = torch.cat([net, pooled, offset], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        offset = offset_code.unsqueeze(1).expand(net.size())
        net = torch.cat([net, pooled, offset], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        offset = offset_code.unsqueeze(1).expand(net.size())
        net = torch.cat([net, pooled, offset], dim=2)

        net = self.block_4(net)
        c = self.fc_c(self.actvn(net))

        return c



class DecoderCBatchNorm(nn.Module):
    ''' Decoder class with CBN for ONet 4D.

    Args:
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned temporal code c
        dim (int): points dimension
        hidden_size (int): hidden dimension
        leaky (bool): whether to use leaky activation
        legacy (bool): whether to use legacy version
    '''

    def __init__(self, dim=3, c_dim=128, out_dim=3,
                 hidden_size=256, leaky=False, legacy=False):
        super().__init__()
        self.dim = dim

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        # self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        # self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        # self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, out_dim, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    # For ONet 4D
    def add_time_axis(self, p, t):
        ''' Adds time axis to points.

        Args:
            p (tensor): points
            t (tensor): time values
        '''
        n_pts = p.shape[1]
        t = t.unsqueeze(1).repeat(1, n_pts, 1)
        p_out = torch.cat([p, t], dim=-1)
        return p_out

    def forward(self, p, c, **kwargs):
        ''' Performs a forward pass through the model.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned temporal code c
        '''
        # if p.shape[-1] != self.dim:
        #     p = self.add_time_axis(p, kwargs['t'])

        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        net = self.block0(net, c)
        net = self.block1(net, c)
        # net = self.block2(net, c)
        # net = self.block3(net, c)
        # net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.transpose(1, 2)

        return out