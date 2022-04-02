import os
import torch
import argparse
import yaml
from lib.dataset.cape_dataset import HumansDataset
from lib.models import H4D
from lib.evaluator import Evaluator
from torch.utils.data import DataLoader



def main(cfg, device):
    print('...Evaluating motion on CAPE test set...')

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

    test_dataset = HumansDataset(
        dataset_folder=cfg['data']['path'],
        split=cfg['data']['test_split'],
        n_sample_points=cfg['data']['num_input_points'],
        length_sequence=cfg['data']['length_sequence'],
        n_files_per_sequence=cfg['data']['n_files_per_sequence'],
        offset_sequence=cfg['data']['offset_sequence'],
        is_eval=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    Evaluator(
        model=model,
        device=device,
        test_loader=test_loader,
        log_dir=out_dir
    ).run()


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 4D model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')

    args = parser.parse_args()

    with open(os.path.join('configs', args.config + '.yaml'), 'r') as f:
        cfg = yaml.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(cfg, device)
