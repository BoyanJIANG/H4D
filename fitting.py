import torch
torch.backends.cudnn.enabled = False
import torch.optim as optim
import numpy as np
import os
from os.path import join
import argparse
import time
import yaml
import trimesh
import datetime
from lib.utils import copy2cpu as c2c
from lib.utils.fitting_utils import SmoothedValue, Prior, get_loss_weights, backward_step
from lib.models import H4D
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.loss.point_mesh_distance import point_face_distance




def compute_p2s_loss(pred_mesh, gt_pcl, seq_len):
    """
    Code taken from: PyTorch3D
    """
    # packed representation for pointclouds
    points = gt_pcl.points_packed()  # (P, 3)
    points_first_idx = gt_pcl.cloud_to_packed_first_idx()
    max_points = gt_pcl.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = pred_mesh.verts_packed()
    faces_packed = pred_mesh.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = pred_mesh.mesh_to_faces_packed_first_idx()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )

    # weight each example by the inverse of number of points in the example
    point_to_cloud_idx = gt_pcl.packed_to_cloud_idx()  # (sum(P_i),)
    num_points_per_cloud = gt_pcl.num_points_per_cloud()  # (N,)
    weights_p = num_points_per_cloud.gather(0, point_to_cloud_idx)
    weights_p = 1.0 / weights_p.float()
    point_to_face = point_to_face * weights_p
    point_dist = point_to_face.sum() / seq_len
    ########

    return point_dist


def back_optim(code_std=0.01, num_iterations=500):

    id_code = torch.ones(1, 10).normal_(mean=0, std=code_std).cuda()
    pose_code = torch.ones(1, 72).normal_(mean=0, std=code_std).cuda()
    motion_code = torch.ones(1, 90).normal_(mean=0, std=code_std).cuda()
    auxiliary_code = torch.ones(1, 128).normal_(mean=0, std=code_std).cuda()

    id_code.requires_grad = True
    pose_code.requires_grad = True
    motion_code.requires_grad = True
    auxiliary_code.requires_grad = True

    optimizer = optim.Adam([id_code, pose_code, motion_code, auxiliary_code], lr=0.03)
    lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

    faces = np.load('assets/smpl_faces.npy')

    gt_pcl = []
    for i in range(seq_len):
        file = join(gt_path, f'{i:04d}.ply')
        mesh = trimesh.load(file, process=False)
        pcl = mesh.sample(8192)
        gt_pcl.append(torch.Tensor(pcl))
    gt_pcl = Pointclouds(points=gt_pcl).to(device)

    prior = Prior(sm=None)['Generic']
    weight_dict = get_loss_weights()
    batch_time = SmoothedValue()

    end = time.time()
    for step in range(num_iterations):
        model.eval()

        _, v1, v2, pred_pose = model.completion(id_code, pose_code,
                                                 motion_code, auxiliary_code)


        faces_tensor = torch.from_numpy(faces.astype(np.int32)).to(device)

        pred_mesh = Meshes(verts=[v2[i] for i in range(seq_len)],
                           faces=[faces_tensor for _ in range(seq_len)])

        loss_dict = dict()

        if args.loss_type == 'p2s':
            # point to surface loss
            loss_dict['p2s'] = compute_p2s_loss(pred_mesh, gt_pcl, seq_len)
        else:
            # chamfer loss
            pred_pcl = sample_points_from_meshes(pred_mesh, 8192)
            loss_dict['chamfer'] = chamfer_distance(pred_pcl, gt_pcl)[0]

        # prior terms borrowed from IPNet (ECCV'20)
        loss_dict['betas'] = torch.mean(id_code ** 2)
        loss_dict['pose_pr'] = prior(pred_pose).mean()

        tot_loss = backward_step(loss_dict, weight_dict, step)

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        lr_sche.step()

        t_batch = time.time() - end
        end = time.time()
        batch_time.update(t_batch)
        eta_seconds = batch_time.global_avg * (num_iterations - step + 1)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        summary_string = f'Step {step + 1} ({step + 1}/{num_iterations}) | ' \
                         f'ETA: {eta_string} | batch_time: {t_batch:.2f} | ' \
                         f'lr:{optimizer.param_groups[0]["lr"]:.6f}'

        for k, v in loss_dict.items():
            if k in ['p2s', 'chamfer']:
                summary_string += f' | {k}: {v:.6f}'
            else:
                summary_string += f' | {k}: {v:.4f}'

        print(summary_string)

        if (step + 1) % 100 == 0:
            # -----------visualization-------------
            if not os.path.exists(mesh_dir):
                os.makedirs(mesh_dir)

            for i in range(seq_len):
                body_mesh = trimesh.Trimesh(vertices=c2c(v1[i]), faces=faces, process=False)
                body_mesh.export(os.path.join(mesh_dir, f'body_{i}.obj'))

                body_mesh = trimesh.Trimesh(vertices=c2c(v2[i]), faces=faces, process=False)
                body_mesh.export(os.path.join(mesh_dir, f'full_{i}.obj'))


            # -----------save codes-------------
            print('Saving latent vectors...')
            torch.save(
                {"it": step+1,
                 "id_code": id_code,
                 "pose_code": pose_code,
                 "motion_code": motion_code,
                 "auxiliary_code": auxiliary_code},
                os.path.join(out_dir, 'latent_vec.pt')
            )

            np.savez(os.path.join(out_dir, 'smpl_params.npz'),
                     beta=id_code.detach().cpu().numpy(),
                     poses=pred_pose.detach().cpu().numpy())


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 4D model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--loss_type', type=str, choices=['chamfer', 'p2s'],
                        default='chamfer', help='Type of the observation loss')

    args = parser.parse_args()

    with open(join('configs', args.config + '.yaml'), 'r') as f:
        cfg = yaml.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = cfg['training']['out_dir']
    seq_len = cfg['data']['length_sequence']
    mesh_dir = join(out_dir, 'back_fitting')

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

    gt_path = './dataset/demo_data/back_fitting'

    back_optim(num_iterations=500)

