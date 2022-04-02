import os
import trimesh
import torch
import pickle
import glob
import numpy as np
from os.path import join
from torch.utils import data
from smplx import SMPL


class HumansDataset(data.Dataset):
    def __init__(self, dataset_folder, split=None, n_sample_points=None, length_sequence=30,
                 n_files_per_sequence=-1, offset_sequence=15, is_eval=False):
        # Attributes
        self.dataset_folder = dataset_folder
        self.length_sequence = length_sequence
        self.n_files_per_sequence = n_files_per_sequence
        self.offset_sequence = offset_sequence
        self.n_sample_points = n_sample_points
        self.is_eval = is_eval

        bm_path = 'assets/SMPL_NEUTRAL.pkl'
        self.bm = SMPL(model_path=bm_path)
        self.faces = np.load('assets/smpl_faces.npy')

        with open(join(self.dataset_folder, 'cape_betas.pkl'), 'rb') as f:
            self.all_beta = pickle.load(f)

        with open(join(self.dataset_folder, 'invalid_start_idx.pkl'), 'rb') as f:
            self.invalid_start_idx = pickle.load(f)

        # Get all models
        if split in ['train', 'val']:
            self.models = []
            if split is not None and os.path.exists(join(self.dataset_folder, split + '.txt')):
                split_file = join(self.dataset_folder, split + '.txt')
                with open(split_file, 'r') as f:
                    models_c = f.read().split('\n')
            else:
                print('Do not find motion list!')
            models_c = list(filter(lambda x: len(x) > 0, models_c))

            subpath = join(self.dataset_folder, 'cape_release', 'sequences')
            models_len = self.get_models_seq_len(subpath, models_c)
            models_c, start_idx = self.subdivide_into_sequences(
                models_c, models_len)
            self.models += [
                {'model': m, 'start_idx': start_idx[i]}
                for i, m in enumerate(models_c)
            ]
            if split == 'val':
                self.models = self.models[:2000]

        elif split == 'test':
            with open(join(self.dataset_folder, 'test_subseqs.pkl'), 'rb') as f:
                self.models = pickle.load(f)

        else:
            print("Invalid split! Please choose from ['train', 'val', 'test'].")


    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        model = self.models[idx]['model']
        start_idx = self.models[idx]['start_idx']
        human_id = model[:5]
        motion_id = model[6:]

        gt_beta = self.all_beta[human_id]
        gt_beta = torch.Tensor(gt_beta)

        files = sorted(glob.glob(join(self.dataset_folder, 'cape_release', 'sequences',
                                      human_id, motion_id, '*.npz')))

        inp_verts = []
        gt_poses = []
        gt_full = []
        gt_offset_cano = []
        v_mini_body_cano = np.load(os.path.join(self.dataset_folder, 'minimal_wo_hair',
                                                human_id + '_minimal_no_hair.npy'))
        for f in files[start_idx: start_idx + self.length_sequence]:
            frame_data = np.load(f)

            # gt poses
            gt_poses.append(frame_data['pose'])

            # gt clothed mesh vertices and input point cloud
            full_verts = frame_data['v_posed'] - frame_data['transl']
            gt_full.append(full_verts)
            mesh = trimesh.Trimesh(vertices=full_verts, faces=self.faces, process=False)
            samples = mesh.sample(self.n_sample_points)
            inp_verts.append(samples)

            # gt offsets under the canonical pose
            v_cloth_cano = frame_data['v_cano']
            clo_disps = v_cloth_cano - v_mini_body_cano
            gt_offset_cano.append(clo_disps)

        inp_verts = torch.Tensor(inp_verts)
        gt_poses = torch.Tensor(gt_poses)
        gt_full = torch.Tensor(gt_full)
        gt_offset_cano = torch.Tensor(gt_offset_cano)

        # gt unclothed body
        gt_smpl = self.bm(
            betas=gt_beta[None].repeat(self.length_sequence, 1),
            body_pose=gt_poses[:, 3:], global_orient=gt_poses[:, :3]
        )

        if not self.is_eval:
            data_dict = {
                'inputs': inp_verts,
                'gt_beta': gt_beta,
                'gt_body': gt_smpl.vertices.detach(),
                'gt_full': gt_full,
                'gt_offset_cano': gt_offset_cano,
                'idx': idx
            }
        else:
            data_dict = {
                'inputs': inp_verts,
                'body_verts': gt_smpl.vertices.detach(),
                'kp_3d': gt_smpl.joints.detach(),
                'idx': idx
            }

        return data_dict


    def get_model_dict(self, idx):
        return self.models[idx]


    def get_models_seq_len(self, subpath, models):
        models_seq_len = []
        for m in models:
            human_id = m[:5]
            motion_id = m[6:]
            models_seq_len.append(len(glob.glob(os.path.join(subpath,
                                                             human_id, motion_id, '*.npz'))))
        return models_seq_len


    def subdivide_into_sequences(self, models, models_len):
        ''' Subdivides model sequence into smaller sequences.

        Args:
            models (list): list of model names
            models_len (list): list of lengths of model sequences
        '''
        length_sequence = self.length_sequence
        n_files_per_sequence = self.n_files_per_sequence
        offset_sequence = self.offset_sequence

        # Remove files before offset
        models_len = [l - offset_sequence for l in models_len]

        # Reduce to maximum number of files that should be considered
        if n_files_per_sequence > 0:
            models_len = [min(n_files_per_sequence, l) for l in models_len]

        models_out = []
        start_idx = []
        for idx, model in enumerate(models):
            for n in range(0, models_len[idx] - length_sequence + 1):
                if (model in self.invalid_start_idx.keys()) and \
                        ((n + offset_sequence) in self.invalid_start_idx[model]):
                    continue
                models_out.append(model)
                start_idx.append(n + offset_sequence)
        return models_out, start_idx