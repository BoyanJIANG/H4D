import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.cape_utils.layers import ChebConv_Coma, Pool

def prepare(device):
    # use the pre-computed mesh pooling matrices
    with open('assets/gcn_dict.pkl', 'rb') as f:
        pkl = pickle.load(f)
    D_t = pkl['D_t']
    U_t = pkl['U_t']
    A_t = pkl['A_t']
    num_nodes = pkl['num_nodes']

    config = {
        'nf': 64,
        'nz': 128,
        'nz_pose': 32,
        'nz_cloth': 8,
        'num_conv_layers': 8,
        'polygon_order': 2
    }

    return config, D_t, U_t, A_t, num_nodes


class Condition(nn.Module):
    def __init__(self, y_dim, nz_cond, nlayers=1):
        super(Condition, self).__init__()

        self.nlayers = nlayers
        if nlayers == 1:
            self.fc1 = nn.Linear(y_dim, nz_cond)
        else:
            if nz_cond < y_dim // 2:
                n_out_fc1 = y_dim // 2
            elif nz_cond < y_dim * 2:
                n_out_fc1 = y_dim
            else:
                n_out_fc1 = nz_cond // 2

            self.fc1 = nn.Linear(y_dim, n_out_fc1)
            self.fc2 = nn.Linear(n_out_fc1, nz_cond)
            self.actvn = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        if self.nlayers == 1:
            return self.fc1(x)
        else:
            x = self.actvn(self.fc1(x))
            x = self.fc2(x)
            return x


class Conv_Res_Block(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_mtx, A_edge_index, A_norm):
        super(Conv_Res_Block, self).__init__()

        # self.pool = Pool()
        self.conv1 = ChebConv_Coma(in_channels, out_channels // 2, K=1, bias=False)
        self.conv2 = ChebConv_Coma(out_channels // 2, out_channels // 2, K=2, bias=False)
        self.conv3 = ChebConv_Coma(out_channels // 2, out_channels, K=1, bias=False)
        self.conv4 = None

        if in_channels != out_channels:
            self.conv4 = ChebConv_Coma(in_channels, out_channels, K=1, bias=False)

        self.gn1 = nn.GroupNorm(32, in_channels)
        self.gn2 = nn.GroupNorm(32, out_channels // 2)
        self.gn3 = nn.GroupNorm(32, out_channels // 2)
        self.upsample_mtx = upsample_mtx
        self.A_edge_index = A_edge_index
        self.A_norm = A_norm

    def forward(self, x):
        device = x.device
        x_unpooled = Pool(x, self.upsample_mtx.to(device))
        x = F.relu(self.gn1(x_unpooled.transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.gn2(self.conv1(x, self.A_edge_index.to(device), self.A_norm.to(device)).transpose(1, 2)).transpose(1, 2))
        x = F.relu(self.gn3(self.conv2(x, self.A_edge_index.to(device), self.A_norm.to(device)).transpose(1, 2)).transpose(1, 2))
        x = self.conv3(x, self.A_edge_index.to(device), self.A_norm.to(device))
        if self.conv4:
            x_unpooled = self.conv4(x_unpooled, self.A_edge_index.to(device), self.A_norm.to(device))
        out = x + x_unpooled
        return out


class CAPE_Decoder(nn.Module):
    def __init__(self, device=None):
        super(CAPE_Decoder, self).__init__()

        config, downsample_matrices, upsample_matrices, \
        adjacency_matrices, num_nodes = prepare(device)

        nf = config['nf']
        self.n_layers = config['num_conv_layers']
        self.filters = [nf, nf, 2*nf, 2*nf, 4*nf, 4*nf, 8*nf, 8*nf]
        self.res_blk_dim = self.filters + [self.filters[-1]]

        self.pose_fc = Condition(14*9, config['nz_pose'], nlayers=2)
        # self.cloth_fc = Condition(4, config['nz_cloth'], nlayers=1)

        out_nodes = int(num_nodes[-1] * self.filters[-1]) // 8
        self.fc1 = nn.Linear(config['nz']+config['nz_pose'], out_nodes)
        self.conv1 = ChebConv_Coma(self.filters[-1] // 8, self.filters[-1], K=1, bias=False)

        self.K = config['polygon_order']
        self.num_nodes = num_nodes
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        self.res_blocks = nn.ModuleList([
            Conv_Res_Block(self.res_blk_dim[-i-1]+config['nz_pose'],
                           self.res_blk_dim[-i-2], self.upsample_matrices[-i-1],
                           self.A_edge_index[-i-2], self.A_norm[-i-2])
            for i in range(self.n_layers)
        ])

        self.conv_out = ChebConv_Coma(self.filters[0]+config['nz_pose'], 3, K=self.K, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(1, 6890, 3).float())


    def forward(self, x, pose):
        '''
        :param x (tensor): (bs, nz)
        :param pose (tensor): (bs, 14*9)
        :return:
        '''
        device = x.device
        bs = x.shape[0]

        y1 = self.pose_fc(pose)
        x = torch.cat([x, y1], -1)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = x.reshape([bs, int(self.num_nodes[-1]), -1])
        x = self.conv1(x, self.A_edge_index[-1].to(device), self.A_norm[-1].to(device))

        cond1 = y1.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([x, cond1], -1)

        for i in range(self.n_layers):
            x = self.res_blocks[i](x)
            cond1 = y1.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, cond1], -1)

        x = self.conv_out(x, self.A_edge_index[0].to(device), self.A_norm[0].to(device))
        out = x + self.out_bias

        return out



class CAPE_Decoder_no_pose(nn.Module):
    def __init__(self, device=None):
        super(CAPE_Decoder_no_pose, self).__init__()

        config, downsample_matrices, upsample_matrices, \
        adjacency_matrices, num_nodes = prepare(device)

        nf = config['nf']
        self.n_layers = config['num_conv_layers']
        self.filters = [nf, nf, 2*nf, 2*nf, 4*nf, 4*nf, 8*nf, 8*nf]
        self.res_blk_dim = self.filters + [512]

        out_nodes = int(num_nodes[-1] * self.filters[-1]) // 8
        self.fc1 = nn.Linear(config['nz'], out_nodes)
        self.conv1 = ChebConv_Coma(64, self.filters[-1], K=1, bias=False)

        self.K = config['polygon_order']
        self.num_nodes = num_nodes
        self.upsample_matrices = upsample_matrices
        self.adjacency_matrices = adjacency_matrices
        self.A_edge_index, self.A_norm = zip(*[ChebConv_Coma.norm(self.adjacency_matrices[i]._indices(),
                                                                  num_nodes[i]) for i in range(len(num_nodes))])
        self.res_blocks = nn.ModuleList([
            Conv_Res_Block(self.res_blk_dim[-i-1],
                           self.res_blk_dim[-i-2], self.upsample_matrices[-i-1],
                           self.A_edge_index[-i-2], self.A_norm[-i-2])
            for i in range(self.n_layers)
        ])

        self.conv_out = ChebConv_Coma(self.filters[0], 3, K=self.K, bias=False)
        self.out_bias = nn.Parameter(torch.zeros(1, 6890, 3).float())


    def forward(self, x):
        '''
        :param x (tensor): (bs, nz)
        :param pose (tensor): (bs, 14*9)
        :return:
        '''
        device = x.device
        bs = x.shape[0]

        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = x.reshape([bs, int(self.num_nodes[-1]), -1])
        x = self.conv1(x, self.A_edge_index[-1].to(device), self.A_norm[-1].to(device))

        for i in range(self.n_layers):
            x = self.res_blocks[i](x)

        x = self.conv_out(x, self.A_edge_index[0].to(device), self.A_norm[0].to(device))
        out = x + self.out_bias

        return out