import numpy as np
from sklearn.decomposition import PCA
from os.path import join
import glob
import torch
from tqdm import tqdm
import argparse
import yaml
from lib.dataset.cape_dataset import HumansDataset

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 4D model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--body_pc', type=int, default=86,
                    help='Number of principal components for body joints rotations')
parser.add_argument('--global_pc', type=int, default=4,
                    help='Number of principal components for global orientation')

args = parser.parse_args()

with open(join('configs', args.config + '.yaml'), 'r') as f:
    cfg = yaml.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = HumansDataset(
    dataset_folder=cfg['data']['path'],
    split=cfg['data']['train_split'],
    n_sample_points=cfg['data']['num_input_points'],
    length_sequence=cfg['data']['length_sequence'],
    n_files_per_sequence=cfg['data']['n_files_per_sequence'],
    offset_sequence=cfg['data']['offset_sequence']
)

models = train_dataset.models

all_poses = []
for m in tqdm(train_dataset.models):
    human_id = m['model'][:5]
    motion_id = m['model'][6:]
    start_idx = m['start_idx']
    files = sorted(glob.glob(join(train_dataset.dataset_folder, 'cape_release', 'sequences',
                                  human_id, motion_id, '*.npz')))
    gt_poses = []
    for f in files[start_idx: start_idx + train_dataset.length_sequence]:
        frame_data = np.load(f)
        gt_poses.append(frame_data['pose'])
    all_poses.append(np.array(gt_poses))
all_poses = np.array(all_poses)

init_pose = np.tile(all_poses[:, 0:1, :], [1, all_poses.shape[1], 1])
delta_poses = all_poses - init_pose  # N, 30, 72

X_body = delta_poses[:, 1:, 3:].reshape(delta_poses.shape[0], -1)
X_global_orient = delta_poses[:, 1:, :3].reshape(delta_poses.shape[0], -1)

pca_body = PCA(n_components=args.body_pc)
x_body = pca_body.fit_transform(X_body)

pca_global = PCA(n_components=args.global_pc)
x_global_orient = pca_global.fit_transform(X_global_orient)

out_dict = {
    'joint_pc': pca_body.components_,
    'joint_mean': pca_body.mean_,
    'rot_pc': pca_global.components_,
    'rot_mean': pca_global.mean_,
}

save_path = 'assets/pca_retrained.npz'
print(f'PCA results are saved in {save_path}.')
np.savez(save_path, **out_dict)

