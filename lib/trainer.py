import os
import torch
import trimesh
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from lib.utils import copy2cpu as c2c
from trimesh.exchange.export import export_mesh



class Trainer(object):
    def __init__(self, model, optimizer, lr_sche, only_stage1, device=None, vis_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.only_stage1 = only_stage1
        self.lr_sche = lr_sche
        self.device = device
        self.vis_dir = vis_dir

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a train step.

        Args:
            data (tensor): training data
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss_dict = self.compute_loss(data)
        loss_dict['loss'].backward()
        self.optimizer.step()
        if not self.only_stage1:
            self.lr_sche.step()
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
        return loss_dict


    def evaluate(self, val_loader, out_dir):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): Pytorch dataloader
        '''
        if self.only_stage1:
            eval_loss = []
            idx = np.random.randint(len(val_loader))
            i = 0
            for data in tqdm(val_loader):
                loss, data, model_output = self.eval_step(data)
                eval_loss.append(loss.cpu().numpy())

                if i == idx:
                    self.visualize(data, model_output, out_dir)
                i += 1

            mean_loss = np.mean(eval_loss)
            return mean_loss
        else:
            eval_loss = []
            stage1_loss = []
            stage2_loss = []
            stage3_loss = []
            idx = np.random.randint(len(val_loader))
            i = 0
            for data in tqdm(val_loader):
                loss, loss_stage1, loss_stage2, loss_stage3, data, model_output = self.eval_step(data)
                eval_loss.append(loss.cpu().numpy())
                stage1_loss.append(loss_stage1.cpu().numpy())
                stage2_loss.append(loss_stage2.cpu().numpy())
                stage3_loss.append(loss_stage3.cpu().numpy())

                if i == idx:
                    self.visualize(data, model_output, out_dir)
                i += 1

            mean_loss = np.mean(eval_loss)
            mean_loss_stage1 = np.mean(stage1_loss)
            mean_loss_stage2 = np.mean(stage2_loss)
            mean_loss_stage3 = np.mean(stage3_loss)
            return mean_loss, mean_loss_stage1, mean_loss_stage2, mean_loss_stage3


    def eval_step(self, data):
        ''' Performs a validation step.

        Args:
            data (tensor): validation data
        '''
        self.model.eval()
        device = self.device
        inputs = data.get('inputs').to(device)
        gt = data.get('gt_body').to(device)
        gt_offset_cano = data.get('gt_offset_cano').to(device)

        batch_size, seq_len, n_pts, _ = inputs.size()
        with torch.no_grad():
            model_output = self.model(inputs)
            if self.only_stage1:
                '''
                --------- Evaluate Stage 1 ---------
                '''
                pred_verts = model_output[2].view(batch_size, seq_len, -1, 3)
                loss = torch.sqrt(torch.sum((gt - pred_verts) ** 2, dim=-1)).mean() * 1000
                return loss, data, model_output

            else:
                '''
                --------- Evaluate all stages ---------
                '''
                pred_verts_0 = model_output[3].view(batch_size, seq_len, -1, 3)
                pred_verts_1 = model_output[4].view(batch_size, seq_len, -1, 3)
                pred_offset_cano = model_output[5].view(batch_size, seq_len, -1, 3)

                loss_stage1 = torch.sqrt(torch.sum((gt - pred_verts_0) ** 2, dim=-1)).mean() * 1000
                loss_stage2 = torch.sqrt(torch.sum((gt - pred_verts_1) ** 2, dim=-1)).mean() * 1000
                loss_stage3 = torch.sqrt(torch.sum((gt_offset_cano - pred_offset_cano) ** 2, dim=-1)).mean() * 1000
                loss = loss_stage2 + loss_stage3
                return loss, loss_stage1, loss_stage2, loss_stage3, data, model_output


    def visualize(self, data, model_output, out_dir):
        ''' Performs a visualization step.

        Args:
            data (tensor): visualization data
        '''
        visual_path = os.path.join(out_dir, 'vis')
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)
        gt_body = data['gt_body'][0]
        gt_full = data['gt_full'][0]
        seq_len = gt_body.shape[0]

        faces = np.load('assets/smpl_faces.npy')

        if self.only_stage1:
            vert = model_output[2].view(seq_len, -1, 3)

            for i, v in enumerate(gt_body):
                body_mesh = trimesh.Trimesh(vertices=c2c(v), faces=faces, process=False)
                export_mesh(body_mesh, os.path.join(visual_path, 'gt_body_%d.obj' % i))

            for i, v in enumerate(vert):
                body_mesh = trimesh.Trimesh(vertices=c2c(v), faces=faces, process=False)
                export_mesh(body_mesh, os.path.join(visual_path, 'pred0_%d.obj' % i))

        else:
            vert_0 = model_output[3].view(seq_len, -1, 3)
            vert_1 = model_output[4].view(seq_len, -1, 3)
            vert_2 = model_output[5].view(seq_len, -1, 3)

            for i in range(seq_len):
                body_mesh = trimesh.Trimesh(vertices=c2c(gt_body[i]), faces=faces, process=False)
                body_mesh.export(os.path.join(visual_path, 'gt_body_%d.obj' % i))

                body_mesh = trimesh.Trimesh(vertices=c2c(gt_full[i]), faces=faces, process=False)
                body_mesh.export(os.path.join(visual_path, 'gt_full_%d.obj' % i))

                body_mesh = trimesh.Trimesh(vertices=c2c(vert_0[i]), faces=faces, process=False)
                body_mesh.export(os.path.join(visual_path, 'pred0_%d.obj' % i))

                body_mesh = trimesh.Trimesh(vertices=c2c(vert_1[i]), faces=faces, process=False)
                body_mesh.export(os.path.join(visual_path, 'pred1_%d.obj' % i))

                body_mesh = trimesh.Trimesh(vertices=c2c(vert_2[i]), faces=faces, process=False)
                body_mesh.export(os.path.join(visual_path, 'pred2_%d.obj' % i))


    def compute_loss(self, data):
        ''' Calculates the loss.
        Args:
            data (tensor): training data
        '''
        device = self.device
        # Encode inputs
        inputs = data.get('inputs').to(device)
        batch_size, seq_len, _, _ = inputs.size()
        gt_body = data.get('gt_body').to(device)
        gt_beta = data.get('gt_beta').to(device)
        gt_offset_cano = data.get('gt_offset_cano').to(device)

        l1_error = nn.L1Loss().to(self.device)

        loss_dict = dict()

        model_output = self.model(inputs)  # batch_size, seq_len, 6890, 3

        if self.only_stage1:
            '''
            --------- Train Stage 1 ---------
            '''
            pred_beta = model_output[0]

            pred_verts = model_output[2].view(batch_size, seq_len, -1, 3)
            vert_loss = l1_error(pred_verts, gt_body)
            beta_loss = torch.norm(gt_beta - pred_beta, dim=-1).mean()

            loss_dict['loss'] = beta_loss + vert_loss
            loss_dict['vert_loss'] = vert_loss
            loss_dict['beta_loss'] = beta_loss

        else:
            '''
            --------- Train all stages ---------
            '''
            pred_beta = model_output[0]

            pred_verts_1 = model_output[4].view(batch_size, seq_len, -1, 3)
            pred_offset_cano = model_output[5].view(batch_size, seq_len, -1, 3)

            vert_loss = l1_error(pred_verts_1, gt_body)
            offset_loss = l1_error(pred_offset_cano, gt_offset_cano)
            beta_loss = torch.norm(gt_beta - pred_beta, dim=-1).mean()

            loss_dict['loss'] = beta_loss + vert_loss + 30 * offset_loss

            loss_dict['beta_loss'] = beta_loss
            loss_dict['vert_loss_1'] = vert_loss
            loss_dict['offset_loss'] = offset_loss

        return loss_dict
