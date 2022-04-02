"""
Code taken from: VIBE: Video Inference for Human Body Pose and Shape Estimation, CVPR'20
"""

import os
import pickle
import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp

import trimesh
from progress.bar import Bar

from lib.utils import move_dict_to_device, AverageMeter

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)

class Evaluator():
    def __init__(
            self,
            test_loader,
            model,
            device=None,
            log_dir=None,
    ):
        self.test_loader = test_loader
        self.model = model
        self.device = device
        self.log_dir = log_dir

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d',
                                                      'target_verts', 'pred_verts'])
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def validate(self):
        self.model.eval()

        start = time.time()

        summary_string = ''

        bar = Bar('Validation', fill='#', max=len(self.test_loader))

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        # ---------------------
        for i, target in enumerate(self.test_loader):
            move_dict_to_device(target, self.device)

            # <=============
            with torch.no_grad():
                inp = target['inputs']

                model_out = self.model.evaluate(inp)

                n_kp = model_out.joints.shape[-2]
                pred_j3d = model_out.joints.view(-1, n_kp, 3).cpu().numpy()
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                pred_verts = model_out.vertices.view(-1, 6890, 3).cpu().numpy()
                target_verts = target['body_verts'].view(-1, 6890, 3).cpu().numpy()

                self.evaluation_accumulators['pred_verts'].append(pred_verts)
                self.evaluation_accumulators['target_verts'].append(target_verts)

                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)
            # =============>

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.test_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            bar.suffix = summary_string
            bar.next()

        bar.finish()

        logger.info(summary_string)

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')

        '''
        # alignment for smpl joints
        '''
        pred_pelvis = pred_j3ds[:, [0], :]
        target_pelvis = target_j3ds[:, [0], :]
        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        # Absolute error (MPJPE)
        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        target_verts = self.evaluation_accumulators['target_verts']

        m2mm = 1000

        pve = np.mean(compute_error_verts(target_verts=target_verts, pred_verts=pred_verts)) * m2mm
        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'pve': pve,
            'accel': accel,
            'accel_err': accel_err
        }

        log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        print(log_str)
        with open(osp.join(self.log_dir, 'eval_results.txt'), 'w') as f:
            f.write(log_str)

    def run(self):
        self.validate()
        self.evaluate()