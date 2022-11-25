import abc
import json
from pathlib import Path
from typing import Any, Optional

import torch
import tqdm
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter


class EpochBasedTrainer(Module, abc.ABC):

    def __init__(self,
                 *,
                 epochs: int,
                 save_interval: int = 1,
                 epochs_per_validation: int = 1,
                 work_dirs: str | Path = 'work_dirs',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 auto_resume: bool = True) -> None:
        super().__init__()

        self._epochs = epochs
        self._save_interval = save_interval
        self._epochs_per_validation = epochs_per_validation
        self._device = device

        self._work_dirs = Path(work_dirs)
        self._ckpt_dirs = self._work_dirs / 'ckpt'
        self._log_dir = self._work_dirs / 'logs'
        self._tb_writer = SummaryWriter(log_dir=self._log_dir)
        with (self._work_dirs / 'hparams.json').open('w',
                                                     encoding='utf8') as f:
            f.write(json.dumps(self.hparams))

        self._current_epoch = 0
        if auto_resume:
            # FIXME
            # put model to cuda before build optimizer
            # https://github.com/pytorch/pytorch/issues/2830
            if torch.cuda.is_available():
                self.to(self._device)
            self._try_resume()

        self._tb_writer.add_text('hyperparameters',
                                 json.dumps(self.hparams, indent=4))

    def train(self) -> None:
        pbar = tqdm.tqdm(total=self._epochs - self._current_epoch)
        prev_ckpt_path: Optional[Path] = self.find_latest_checkpoint()
        latest_ckpt_path = self._ckpt_dirs / 'latest.pth'

        while self._current_epoch < self._epochs:
            self._current_epoch += 1

            self.before_train_epoch()
            self.train_epoch()
            self.after_train_epoch()

            if self._current_epoch == 1 or \
                    self._current_epoch % self._epochs_per_validation == 0:
                self.validation()

            if self._current_epoch % self._save_interval == 0:
                self.save_checkpoint(ckpt_path=self._ckpt_dirs /
                                     f'epoch={self._current_epoch}.pth')

            if prev_ckpt_path is not None:
                prev_ckpt_path.unlink()
            prev_ckpt_path = \
                self._ckpt_dirs / f'epoch={self._current_epoch}.pth'
            self.save_checkpoint(ckpt_path=prev_ckpt_path)

            latest_ckpt_path.unlink(missing_ok=True)
            latest_ckpt_path.symlink_to(prev_ckpt_path)

            pbar.update(1)

        self.test()

    @abc.abstractmethod
    def before_train_epoch(self) -> None:
        ...

    @abc.abstractmethod
    def train_epoch(self) -> None:
        ...

    @abc.abstractmethod
    def after_train_epoch(self) -> None:
        ...

    @abc.abstractmethod
    def validation(self) -> None:
        ...

    @abc.abstractmethod
    def test(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def hparams(self) -> dict:
        ...

    @property
    def stats(self) -> dict[str, Any]:
        return dict(hyperparameters=self.hparams,
                    runtime_stats=self._current_epoch)

    def save_checkpoint(self, ckpt_path: str | Path) -> None:
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
        if not ckpt_path.parent.exists():
            ckpt_path.parent.mkdir(parents=True)

        save_obj = dict(stats=self.stats, state_dict=self.state_dict())
        torch.save(save_obj, ckpt_path)

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        if isinstance(ckpt_path, str):
            ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise ValueError(f'Path `{ckpt_path}` does not exist')

        print(f'Load from checkpoint: {ckpt_path}')

        load_obj = torch.load(ckpt_path, map_location='cpu')
        load_stats = load_obj['stats']
        if (old_hp := load_stats['hyperparameters']) != self._hparams:
            print(f'Hyperparameters not same.\n '
                  f'Previous: {old_hp} \n'
                  f'new: {self._hparams}')

        self._current_epoch = load_stats['runtime_stats']
        self.load_state_dict(load_obj['state_dict'])

    def find_latest_checkpoint(self) -> Optional[Path]:
        ckpt_dir = self._ckpt_dirs

        ckpt_path_list = [
            x for x in ckpt_dir.glob('*.pth') if x.name.startswith('latest')
        ]
        if len(ckpt_path_list) == 0:
            print('Unable to find checkpoint')
            return None

        latest_ckpt_path = max(ckpt_path_list,
                               key=lambda x: int(x.stem.split('=')[1]))

        return latest_ckpt_path

    def _try_resume(self) -> None:
        latest_ckpt_path = self.find_latest_checkpoint()
        if latest_ckpt_path is not None:
            self.load_checkpoint(latest_ckpt_path)
