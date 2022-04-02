import os
import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np
import argparse
import time
import yaml
from os.path import join
from torch.utils.data import DataLoader
from lib.dataset.cape_dataset import HumansDataset
from lib.dataset import core as data
from lib.models import H4D
from lib.trainer import Trainer



# Arguments
parser = argparse.ArgumentParser(
    description='Train a 4D model.'
)
parser.add_argument('config', type=str, help='Path to config file.')

args = parser.parse_args()

with open(join('configs', args.config + '.yaml'), 'r') as f:
    cfg = yaml.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.config == 'h4d_stage1':
    train_stage = 1
    only_stage1 = True
    assert cfg['model']['use_aux_code'] == 0
else:
    train_stage = 2
    only_stage1 = False
    assert cfg['model']['use_aux_code'] == 1

print(f'Start the training of Stage-{train_stage}')


# Shorthands
out_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
batch_size_vis = cfg['training']['batch_size_vis']
batch_size_val = cfg['training']['batch_size_val']
backup_every = cfg['training']['backup_every']
lr = cfg['training']['learning_rate']

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(os.path.join(out_dir, 'train_cfg.yaml'), 'w') as f:
    yaml.dump(cfg, f)

# Dataset
train_dataset = HumansDataset(
    dataset_folder=cfg['data']['path'],
    split=cfg['data']['train_split'],
    n_sample_points=cfg['data']['num_input_points'],
    length_sequence=cfg['data']['length_sequence'],
    n_files_per_sequence=cfg['data']['n_files_per_sequence'],
    offset_sequence=cfg['data']['offset_sequence']
)

val_dataset = HumansDataset(
    dataset_folder=cfg['data']['path'], split=cfg['data']['val_split'],
    n_sample_points=cfg['data']['num_input_points'],
    length_sequence=cfg['data']['length_sequence'],
    n_files_per_sequence=cfg['data']['n_files_per_sequence'],
    offset_sequence=cfg['data']['offset_sequence'])

# Dataloader
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=8, shuffle=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size_val, num_workers=8, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)


# Model
model = nn.DataParallel(
    H4D(aux_code=cfg['model']['use_aux_code'],
        device=device).to(device)
)

if train_stage == 2:
    # --------------load the pretrained model of Stage-1----------------
    print('Loading model params...')
    model_dict = model.state_dict()
    load_modules = ['id_encoder', 'pose_encoder', 'point_encoder', 'motion_encoder']
    data = torch.load('out/h4d_stage1/model_best.pt', map_location=device)
    params = data['model']
    load_dict = {k: v for k, v in params.items() if k.split('.')[1] in load_modules}
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    # -------------------------------------------------------------

optimizer = optim.Adam(model.parameters(), lr=lr)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training']['lr_decay_step'],
                                    gamma=cfg['training']['lr_decay']) if not only_stage1 else None

trainer = Trainer(model, optimizer, lr_sche, only_stage1, device, out_dir)

try:
    filename = join(out_dir, 'model.pt')
    load_dict = torch.load(filename)
    print(filename)
    print('=> Loading checkpoint from local file...')
    model.load_state_dict(load_dict['model'])
except FileNotFoundError:
    load_dict = dict()

epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

t1 = t2 = t3 = 0
while True:
    epoch_it += 1

    for batch in train_loader:

        t1 = time.time()

        for _ in range(5):
            it += 1
            loss_dict = trainer.train_step(batch)

        t2 = time.time()

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print_string = f'Epoch {epoch_it} | iter {it} | '

            for k, v in loss_dict.items():
                logger.add_scalar(f'train/{k}', v, it)
                print_string += f'{k}={v:.4f} | '

            print_string += f't_calculate={(t2 - t1):.4f} | t_data={(t1 - t3):.4f}'

            print(print_string)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            torch.save({'model': model.state_dict(),
                        'epoch_it': epoch_it,
                        'it': it,
                        'loss_val_best': metric_val_best},
                       join(out_dir, 'model.pt'))

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            torch.save({'model': model.state_dict(),
                        'epoch_it': epoch_it,
                        'it': it,
                        'loss_val_best': metric_val_best},
                       join(out_dir, 'model_%d.pt' % it))

        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            if not only_stage1:
                metric_val, metric_stage1, metric_stage2, metric_stage3 = trainer.evaluate(val_loader, out_dir)
                print('Validation metric (%s): %.4f'
                      % (model_selection_metric, metric_val))
                print('Validation metric stage1: %.4f'
                      % (metric_stage1))
                print('Validation metric stage2: %.4f'
                      % (metric_stage2))
                print('Validation metric stage3: %.4f'
                      % (metric_stage3))

                logger.add_scalar('val/%s' % model_selection_metric, metric_val, it)
                logger.add_scalar('val/stage1', metric_stage1, it)
                logger.add_scalar('val/stage2', metric_stage2, it)
                logger.add_scalar('val/stage3', metric_stage3, it)
            else:
                metric_val = trainer.evaluate(val_loader, out_dir)
                print('Validation metric (%s): %.4f'
                      % (model_selection_metric, metric_val))

                logger.add_scalar('val/%s' % model_selection_metric, metric_val, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (%s): %.4f' % (model_selection_metric, metric_val_best))
                torch.save({'model': model.state_dict(),
                            'epoch_it': epoch_it,
                            'it': it,
                            'loss_val_best': metric_val_best},
                           join(out_dir, 'model_best.pt'))

            print('Current best validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val_best))

        t3 = time.time()
