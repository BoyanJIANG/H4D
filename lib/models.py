import sys
sys.path.append('../')
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from smplx import SMPL
from smplx.lbs import batch_rodrigues
from lib.pointnet import SimplePointnet, ResnetPointnet
from lib.cape_utils.model_cape import CAPE_Decoder
from lib.cape_utils.utils import filter_cloth_pose


BM_PATH = 'assets/SMPL_NEUTRAL.pkl'
PCA_DICT = 'assets/pca_pretrained.npz'

# --------------------------------------------------------
# filter out vertices of unclothed parts, borrow from CAPE (Qianli et al. CVPR2020)
part_segm_filepath = 'assets/smpl_vert_segmentation.json'
part_segm = json.load(open(part_segm_filepath))
unclothed_part = part_segm['neck'] + \
                 part_segm['leftHand'] + part_segm['leftHandIndex1'] + \
                 part_segm['rightHand'] + part_segm['rightHandIndex1'] + \
                 part_segm['leftFoot'] + part_segm['leftToeBase'] + \
                 part_segm['rightFoot'] + part_segm['rightToeBase']
useful_verts_idx = np.array(list(filter(lambda x: x not in unclothed_part, range(6890))))
# --------------------------------------------------------


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            input_size=1024,
            hidden_size=1024,
            out_size=128,
            add_linear=True,
            bidirectional=False
    ):
        super(TemporalEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=n_layers
        )

        self.linear = None
        if bidirectional:
            self.linear = nn.Linear(hidden_size*2, out_size)
        elif add_linear:
            self.linear = nn.Linear(hidden_size, out_size)
        self.out_size = out_size

    def forward(self, x):
        n, t, f = x.shape
        x = x.permute(1, 0, 2)  # NTF -> TNF
        self.gru.flatten_parameters()
        y, _ = self.gru(x)
        if self.linear:
            y = F.relu(y)
            y = self.linear(y.view(-1, y.size(-1)))
            y = y.view(t, n, self.out_size)
        y = y.permute(1, 0, 2)  # TNF -> NTF
        return y



class H4D(nn.Module):
    def __init__(self, point_feat_dim=128, id_dim=10, pose_dim=72, motion_dim=90,
                 aux_dim=128, z_cloth_dim=128, hidden_dim=128, aux_code=True, device=None):
        super().__init__()

        # --------- Encoders ---------
        self.id_encoder = ResnetPointnet(c_dim=id_dim, hidden_dim=hidden_dim)
        self.pose_encoder = ResnetPointnet(c_dim=pose_dim, hidden_dim=hidden_dim)
        self.point_encoder = SimplePointnet(c_dim=point_feat_dim, hidden_dim=128)
        self.motion_encoder = TemporalEncoder(input_size=point_feat_dim, add_linear=True,
                                              hidden_size=512, out_size=motion_dim)

        # --------- Load principal components ---------
        pca_dict = np.load(PCA_DICT)
        for k, v in pca_dict.items():
            self.register_buffer(k, torch.from_numpy(v).float().to(device))

        # --------- SMPL Relates ---------
        self.pose_dim = pose_dim
        self.bm = SMPL(model_path=BM_PATH,
                       create_transl=False)
        self.cloth_bm = SMPL(model_path=BM_PATH,
                             create_transl=False)
        self.v_template = self.cloth_bm.v_template.clone()

        # --------- Auxiliary Decoder ---------
        self.aux_code = aux_code
        self.aux_dim = aux_dim
        self.z_cloth_dim = z_cloth_dim
        if self.aux_code:
            self.aux_encoder = TemporalEncoder(input_size=point_feat_dim, add_linear=True,
                                               hidden_size=512, out_size=aux_dim)
            self.aux_decoder = TemporalEncoder(
                input_size=motion_dim + aux_dim + pose_dim,
                hidden_size=512,
                add_linear=True,
                out_size=pose_dim)

            self.cloth_gru = TemporalEncoder(input_size=14*9 + aux_dim,
                                              add_linear=True,
                                              hidden_size=512,
                                              out_size=z_cloth_dim)
            self.offset_decoder = CAPE_Decoder(device=device)


    def completion(self, c_i, c_p, c_m, c_a):
        device = c_i.device
        bm = self.bm.to(device)
        batch_size = 1
        seq_len = 30

        c_i_batch = c_i.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size*seq_len, -1)

        delta_root_orient = torch.matmul(c_m[:, :4], self.rot_pc) + self.rot_mean
        delta_root_orient = delta_root_orient.view(batch_size, seq_len - 1, -1)
        delta_body_pose = torch.matmul(c_m[:, 4:], self.joint_pc) + self.joint_mean
        delta_body_pose = delta_body_pose.view(batch_size, seq_len - 1, -1)

        delta_pose = torch.cat([delta_root_orient, delta_body_pose], -1)
        poses = c_p.unsqueeze(1) + delta_pose   # B, 29, 156
        poses_stage1 = torch.cat([c_p.view(batch_size, 1, -1), poses], 1)   # B, 30, 156

        pred0 = bm(
            betas=c_i_batch,
            body_pose=poses_stage1.view(-1, self.pose_dim)[:, 3:],
            global_orient=poses_stage1.view(-1, self.pose_dim)[:, :3],
        )

        if self.aux_code:
            concat_feat = torch.cat([
                c_m.unsqueeze(1).repeat(1, seq_len, 1),
                c_a.unsqueeze(1).repeat(1, seq_len, 1),
                poses_stage1
            ], -1)
            aux_delta_pose = self.aux_decoder(concat_feat).view_as(poses_stage1)  # B, 30, 156
            final_poses = poses_stage1 + aux_delta_pose
            final_poses = final_poses.view(-1, self.pose_dim)
            pred1 = bm(
                betas=c_i_batch,
                body_pose=final_poses[:, 3:],
                global_orient=final_poses[:, :3],
                return_full_pose=True
            )

            axis_angle_vecs = final_poses.reshape(-1, 3)
            rots = batch_rodrigues(axis_angle_vecs).reshape(-1, 216)  # bs*30, 216
            filtered_rots = filter_cloth_pose(rots)  # bs*30, 14*9
            z_cloth = self.cloth_gru(torch.cat([
                c_a.unsqueeze(1).repeat(1, seq_len, 1),
                filtered_rots.reshape(batch_size, seq_len, -1),
            ], -1))
            pred_offset = self.offset_decoder(z_cloth.reshape(-1, self.aux_dim),
                                              filtered_rots).reshape(batch_size, seq_len, 6890, 3)

            cloth_bm = self.cloth_bm.to(device)
            cloth_bm.v_template = self.v_template.to(device) + pred_offset.reshape(-1, 6890, 3)
            final_verts = cloth_bm(betas=c_i_batch,
                            body_pose=final_poses[:, 3:],
                            global_orient=final_poses[:, :3],).vertices

            return pred0.vertices, pred1.vertices, final_verts, pred1.full_pose
        else:
            return pred0.vertices


    def motion_retargeting(self, id_seq, motion_seq):
        device = id_seq.device
        batch_size, seq_len, n_pts, _ = id_seq.shape

        c_i_1 = self.id_encoder(id_seq[:, 0, :])  # B, 16
        point_feat_1 = torch.stack(
            [self.point_encoder(id_seq[i])
             for i in range(batch_size)], dim=0)
        c_a_1 = self.aux_encoder(point_feat_1)[:, -1]  # B, 128

        c_p_2 = self.pose_encoder(motion_seq[:, 0, :])  # B, 156
        point_feat_2 = torch.stack(
            [self.point_encoder(motion_seq[i])
             for i in range(batch_size)], dim=0)
        c_m_2 = self.motion_encoder(point_feat_2)[:, -1]  # B, 6 + 64
        c_a_2 = self.aux_encoder(point_feat_2)[:, -1]  # B, 128

        c_i = c_i_1
        c_i_batch = c_i.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size*seq_len, -1)
        c_p = c_p_2
        c_m = c_m_2

        delta_root_orient = torch.matmul(c_m[:, :4], self.rot_pc) + self.rot_mean
        delta_root_orient = delta_root_orient.view(batch_size, seq_len - 1, -1)
        delta_body_pose = torch.matmul(c_m[:, 4:], self.joint_pc) + self.joint_mean
        delta_body_pose = delta_body_pose.view(batch_size, seq_len - 1, -1)

        delta_pose = torch.cat([delta_root_orient, delta_body_pose], -1)
        poses = c_p.unsqueeze(1) + delta_pose  # B, 29, 156
        poses_stage1 = torch.cat([c_p.view(batch_size, 1, -1), poses], 1)  # B, 30, 156

        concat_feat = torch.cat([
            c_m.unsqueeze(1).repeat(1, seq_len, 1),
            c_a_2.unsqueeze(1).repeat(1, seq_len, 1),
            poses_stage1
        ], -1)
        aux_delta_pose = self.aux_decoder(concat_feat).view_as(poses_stage1)  # B, 30, 156
        final_poses = poses_stage1 + aux_delta_pose
        final_poses = final_poses.view(-1, self.pose_dim)

        axis_angle_vecs = final_poses.reshape(-1, 3)
        rots = batch_rodrigues(axis_angle_vecs).reshape(-1, 216)  # bs*30, 216
        filtered_rots = filter_cloth_pose(rots)  # bs*30, 14*9
        z_cloth = self.cloth_gru(torch.cat([
            c_a_1.unsqueeze(1).repeat(1, seq_len, 1),
            filtered_rots.reshape(batch_size, seq_len, -1),
        ], -1))
        pred_offset = self.offset_decoder(z_cloth.reshape(-1, self.aux_dim),
                                          filtered_rots).reshape(batch_size, seq_len, 6890, 3)

        cloth_bm = self.cloth_bm.to(device)
        cloth_bm.v_template = self.v_template.to(device) + pred_offset.reshape(-1, 6890, 3)
        final_verts = cloth_bm(betas=c_i_batch,
                               body_pose=final_poses[:, 3:],
                               global_orient=final_poses[:, :3], ).vertices

        return final_verts


    def forward(self, inputs):
        bm = self.bm.to(inputs)
        batch_size, seq_len, n_pts, _ = inputs.shape

        c_i = self.id_encoder(inputs[:, 0, :])  # B, 10
        c_i_batch = c_i.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size*seq_len, -1)
        c_p = self.pose_encoder(inputs[:, 0, :])  # B, 72

        point_feat = torch.stack(
            [self.point_encoder(inputs[i])
             for i in range(batch_size)], dim=0)
        c_m = self.motion_encoder(point_feat)[:, -1]  # B, 6+84

        delta_root_orient = torch.matmul(c_m[:, :4], self.rot_pc) + self.rot_mean
        delta_root_orient = delta_root_orient.view(batch_size, seq_len - 1, -1)
        delta_body_pose = torch.matmul(c_m[:, 4:], self.joint_pc) + self.joint_mean
        delta_body_pose = delta_body_pose.view(batch_size, seq_len - 1, -1)

        delta_pose = torch.cat([delta_root_orient, delta_body_pose], -1)
        poses = c_p.unsqueeze(1) + delta_pose  # B, 29, 156
        poses_stage1 = torch.cat([c_p.view(batch_size, 1, -1), poses], 1)  # B, 30, 156

        pred0 = bm(
            betas=c_i_batch,
            body_pose=poses_stage1.view(-1, self.pose_dim)[:, 3:],
            global_orient=poses_stage1.view(-1, self.pose_dim)[:, :3],
        ).vertices

        if self.aux_code:
            '''compensate pose'''
            c_a = self.aux_encoder(point_feat)[:, -1]  # B, 128
            concat_feat = torch.cat([
                c_m.unsqueeze(1).repeat(1, seq_len, 1),
                c_a.unsqueeze(1).repeat(1, seq_len, 1),
                poses_stage1
            ], -1)
            aux_delta_pose = self.aux_decoder(concat_feat).view_as(poses_stage1)  # B, 30, 156
            final_poses = poses_stage1 + aux_delta_pose
            final_poses = final_poses.view(-1, self.pose_dim)
            pred1 = bm(
                betas=c_i_batch,
                body_pose=final_poses[:, 3:],
                global_orient=final_poses[:, :3],
                return_full_pose=True
            ).vertices

            '''compensate shape'''
            axis_angle_vecs = final_poses.reshape(-1, 3)
            rots = batch_rodrigues(axis_angle_vecs).reshape(-1, 216)  # bs*30, 216
            filtered_rots = filter_cloth_pose(rots)  # bs*30, 14*9
            z_cloth = self.cloth_gru(torch.cat([
                c_a.unsqueeze(1).repeat(1, seq_len, 1),
                filtered_rots.reshape(batch_size, seq_len, -1),
            ], -1))
            pred_offset = self.offset_decoder(z_cloth.reshape(-1, self.z_cloth_dim),
                                              filtered_rots).reshape(batch_size, seq_len, 6890, 3)

            '''apply linear blend skinning'''
            cloth_bm = self.cloth_bm.to(inputs)
            cloth_bm.v_template = self.v_template.to(inputs) + pred_offset.reshape(-1, 6890, 3)
            final_posed_vert = cloth_bm(betas=c_i_batch,
                                        body_pose=final_poses[:, 3:],
                                        global_orient=final_poses[:, :3],).vertices

            return c_i, c_m, c_a, pred0, pred1, pred_offset, final_posed_vert
        else:
            return c_i, c_m, pred0


    def evaluate(self, inputs):
        bm = self.bm.to(inputs)
        batch_size, seq_len, n_pts, _ = inputs.shape

        c_i = self.id_encoder(inputs[:, 0, :])  # B, 10
        c_i_batch = c_i.unsqueeze(1).repeat(1, seq_len, 1).view(batch_size*seq_len, -1)
        c_p = self.pose_encoder(inputs[:, 0, :])  # B, 72

        point_feat = torch.stack(
            [self.point_encoder(inputs[i])
             for i in range(batch_size)], dim=0)
        c_m = self.motion_encoder(point_feat)[:, -1]  # B, 6+84

        delta_root_orient = torch.matmul(c_m[:, :4], self.rot_pc) + self.rot_mean
        delta_root_orient = delta_root_orient.view(batch_size, seq_len - 1, -1)
        delta_body_pose = torch.matmul(c_m[:, 4:], self.joint_pc) + self.joint_mean
        delta_body_pose = delta_body_pose.view(batch_size, seq_len - 1, -1)

        delta_pose = torch.cat([delta_root_orient, delta_body_pose], -1)
        poses = c_p.unsqueeze(1) + delta_pose  # B, 29, 156
        poses_stage1 = torch.cat([c_p.view(batch_size, 1, -1), poses], 1)  # B, 30, 156

        '''compensate pose'''
        c_a = self.aux_encoder(point_feat)[:, -1]  # B, 128
        concat_feat = torch.cat([
            c_m.unsqueeze(1).repeat(1, seq_len, 1),
            c_a.unsqueeze(1).repeat(1, seq_len, 1),
            poses_stage1
        ], -1)
        aux_delta_pose = self.aux_decoder(concat_feat).view_as(poses_stage1)  # B, 30, 156
        final_poses = poses_stage1 + aux_delta_pose
        final_poses = final_poses.view(-1, self.pose_dim)
        pred1 = bm(
            betas=c_i_batch,
            body_pose=final_poses[:, 3:],
            global_orient=final_poses[:, :3],
            return_full_pose=True
        )

        return pred1


