import torch
import numpy as np
import os
import argparse
import yaml
import trimesh
from os.path import join
from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.dataset.cape_dataset import HumansDataset
from lib.utils import copy2cpu as c2c
from smplx import SMPL
from lib.models import H4D


# Arguments
parser = argparse.ArgumentParser(
    description='Train a 4D model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--is_demo', action='store_true', help='whether to run the demo data')


args = parser.parse_args()

with open(join('configs', args.config + '.yaml'), 'r') as f:
    cfg = yaml.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shorthands
out_dir = cfg['training']['out_dir']
if args.is_demo:
    gen_dir = join(out_dir, 'demo_generation')
else:
    gen_dir = join(out_dir, 'generation')
    # Dataset
    dataset = HumansDataset(
        dataset_folder=cfg['data']['path'],
        split=cfg['data']['test_split'],
        n_sample_points=cfg['data']['num_input_points'],
        length_sequence=cfg['data']['length_sequence'],
        n_files_per_sequence=cfg['data']['n_files_per_sequence'],
        offset_sequence=cfg['data']['offset_sequence'])
    # Dataloader
    test_loader = DataLoader(
        dataset, batch_size=1, num_workers=1, shuffle=False)

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


model_path = 'assets/SMPL_NEUTRAL.pkl'
bm = SMPL(model_path=model_path, batch_size=1).to(device)
faces = bm.faces
v_template = bm.v_template.clone()

if args.is_demo:
    # load the input point cloud sequences
    data_path = './dataset/demo_data/4d_reconstruction'
    save_path = join(out_dir, 'demo_generation')
    num_input_points = cfg['data']['num_input_points']
    seq_len = cfg['data']['length_sequence']
    inp_verts = []
    for i in range(seq_len):
        mesh = trimesh.load(join(data_path, f'{i:04d}.ply'), process=False)
        inp_verts.append(mesh.sample(num_input_points))
    inp_verts = torch.Tensor(inp_verts).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        model_output = model(inp_verts)
    final_verts = model_output[-1]
    faces = np.load('assets/smpl_faces.npy')
    for i, v in enumerate(c2c(final_verts)):
        mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
        mesh.export(os.path.join(gen_dir, f'pred_{i}.obj'))
    print('Saving mesh to ', gen_dir)
else:
    for data in tqdm(test_loader):
        idx = data['idx']
        inputs = data['inputs'].to(device)
        gt = data['gt_body'].squeeze()
        gt_full = data['gt_full'].squeeze()
        model_dict = dataset.get_model_dict(idx)

        model.eval()
        with torch.no_grad():
            model_output = model(inputs)

        pred0 = model_output[3]
        pred1 = model_output[4]
        pred2 = model_output[-1]

        save_path = join(gen_dir, model_dict['model'], str(model_dict['start_idx']))
        os.makedirs(save_path, exist_ok=True)

        for i in range(gt.shape[0]):
            body_mesh = trimesh.Trimesh(vertices=c2c(pred0[i]), faces=faces, process=False)
            body_mesh.export(os.path.join(save_path, 'pred0_%d.obj' % i))

            body_mesh = trimesh.Trimesh(vertices=c2c(pred1[i]), faces=faces, process=False)
            body_mesh.export(os.path.join(save_path, 'pred1_%d.obj' % i))

            body_mesh = trimesh.Trimesh(vertices=c2c(pred2[i]), faces=faces, process=False)
            body_mesh.export(os.path.join(save_path, 'pred2_%d.obj' % i))

            body_mesh = trimesh.Trimesh(vertices=c2c(gt[i]), faces=faces, process=False)
            body_mesh.export(os.path.join(save_path, 'gt_body_%d.obj' % i))

            body_mesh = trimesh.Trimesh(vertices=c2c(gt_full[i]), faces=faces, process=False)
            body_mesh.export(os.path.join(save_path, 'gt_full_%d.obj' % i))