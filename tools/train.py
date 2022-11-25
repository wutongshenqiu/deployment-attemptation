from pathlib import Path

import torch
from gpu_helper import GpuHelper

from animal_classification.config import Config, DictAction
from animal_classification.trainer import build_trainer

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config', type=Path, help='Path of config file')
    parser.add_argument('--work_dirs',
                        type=Path,
                        default='work_dirs',
                        help='Path of work directory')
    parser.add_argument('--auto-select-gpu',
                        '-asg',
                        action='store_true',
                        help='Auto select and wait for gpu')
    parser.add_argument('--asg-params',
                        '-asgp',
                        nargs='+',
                        action=DictAction,
                        help='Parameters of gpu helper')

    args = parser.parse_args()
    print(args)

    if args.auto_select_gpu:
        print('Enable auto select gpu')
        if args.asg_params is not None:
            asg_params = args.asg_params
        else:
            asg_params = {}
        gpu_helper = GpuHelper(**asg_params)
        available_indices = gpu_helper.wait_for_available_indices()
        print(f'Find available gpu indices: {available_indices}')
        torch.cuda.set_device(available_indices[0])

    config_path: Path = args.config
    config = Config.fromfile(config_path)

    trainer_config = config.trainer
    work_dirs: Path = args.work_dirs / config_path.stem
    if not work_dirs.exists():
        work_dirs.mkdir(parents=True)
    trainer_config.work_dirs = str(work_dirs)
    print(config.pretty_text)

    trainer = build_trainer(trainer_config)
    print(trainer._nos.optimizer.state_dict().keys())
    print('=' * 500)
    ckpt_path = 'work_dirs/mbv2_animal5_bs32/ckpt/epoch=20.pth'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    print(state_dict['state_dict']['_nos.optimizer'].keys())
    trainer.load_checkpoint('work_dirs/mbv2_animal5_bs32/ckpt/epoch=20.pth')
    torch.save(trainer._nos.network.state_dict(),
               'work_dirs/mbv2_animal5_bs32/ckpt/network.pth')
    exit()
    trainer.train()

    # empty pinned memory
    torch.cuda.empty_cache()
