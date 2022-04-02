import torch
import numpy as np
import os
import argparse
import time
import yaml
import trimesh
from os.path import join
from lib.models import H4D
from lib.utils import copy2cpu as c2c



# Arguments
parser = argparse.ArgumentParser(
    description='Train a 4D model.'
)
parser.add_argument('config', type=str, help='Path to config file.')


args = parser.parse_args()

with open(join('configs', args.config + '.yaml'), 'r') as f:
    cfg = yaml.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
gen_dir = join(out_dir, 'motion_retargeting')

# Output directory
if not os.path.exists(gen_dir):
    os.makedirs(gen_dir)

####################################
# Model
model = H4D(aux_code=cfg['model']['use_aux_code'], device=device).to(device)

# load model weights
out_dir = cfg['training']['out_dir']
filename = os.path.join(out_dir, 'model_best.pt')
load_dict = torch.load(filename, map_location=device)
print(filename)
print('=> Loading checkpoint from local file...')
model_dict = model.state_dict()
params = load_dict['model']
load_params = {k[7:]: v for k, v in params.items()}
model_dict.update(load_params)
model.load_state_dict(model_dict)
####################################

# load input point cloud sequences
data_path = './dataset/demo_data/motion_retargeting'
num_input_points = cfg['data']['num_input_points']
seq_len = cfg['data']['length_sequence']

id_seq = []
motion_seq = []
for i in range(seq_len):
    id_mesh = trimesh.load(join(data_path, 'identity_seq', f'{i:04d}.ply'), process=False)
    id_seq.append(id_mesh.sample(num_input_points))
    motion_mesh = trimesh.load(join(data_path, 'motion_seq', f'{i:04d}.ply'), process=False)
    motion_seq.append(motion_mesh.sample(num_input_points))
id_seq = torch.Tensor(id_seq).unsqueeze(0).to(device)
motion_seq = torch.Tensor(motion_seq).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    verts = model.motion_retargeting(id_seq, motion_seq)

faces = np.load('assets/smpl_faces.npy')
for i, v in enumerate(c2c(verts)):
    mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    mesh.export(os.path.join(gen_dir, f'pred_{i}.obj'))

print('Saving mesh to ', gen_dir)

