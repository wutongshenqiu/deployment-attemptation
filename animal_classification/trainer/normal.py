import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from animal_classification.component import build_nos
from animal_classification.component.nos import SimpleNOS
from animal_classification.data import build_dataloader
from animal_classification.utils import (
    AverageMeter,
    calc_batch_acc,
    collect_hyperparameters,
    evaluate_accuracy,
)

from .base import EpochBasedTrainer


class EpochNormalTrainer(EpochBasedTrainer):

    def __init__(self, *, nos: dict, train_dataloader: dict,
                 val_dataloader: dict, test_dataloader: dict,
                 **kwargs) -> None:
        self._hparams = collect_hyperparameters()

        super().__init__(**kwargs)

        self._nos = build_nos(nos)
        self._train_dataloader = build_dataloader(train_dataloader)
        self._val_dataloader = build_dataloader(val_dataloader)
        self._test_dataloader = build_dataloader(test_dataloader)

    def before_train_epoch(self) -> None:
        ...

    def train_epoch(self) -> None:
        tag_scalar_dict = self._train_network(
            nos=self._nos,
            dataloader=self._train_dataloader,
            device=self._device)
        self._tb_writer.add_scalars(
            f'{type(self._nos.network).__name__} training '
            f'on {type(self._train_dataloader.dataset).__name__}',
            tag_scalar_dict, self._current_epoch)

    def after_train_epoch(self) -> None:
        # Step schedulers
        if (n_scheduler := self._nos.scheduler) is not None:
            n_scheduler.step()

        # log learning rate
        network_lr = {
            type(self._nos.network).__name__.lower():
            self._nos.optimizer.param_groups[0]['lr']
        }
        self._tb_writer.add_scalars('Learning rate', {**network_lr},
                                    self._current_epoch)

    def after_train(self) -> None:
        torch.save(self._nos.network.state_dict(),
                   self._ckpt_dirs / 'network.pth')

    def validation(self) -> None:
        # acc of clean test data on teacher & student
        tag_scalar_dict = evaluate_accuracy(model=self._nos.network,
                                            dataloader=self._val_dataloader,
                                            device=self._device,
                                            top_k_list=(1, 5))
        self._tb_writer.add_scalars(
            f'{type(self._nos.network).__name__} validation on '
            f'{type(self._val_dataloader.dataset).__name__}', tag_scalar_dict,
            self._current_epoch)

    def test(self) -> None:
        # acc of clean test data on teacher & student
        tag_scalar_dict = evaluate_accuracy(model=self._nos.network,
                                            dataloader=self._test_dataloader,
                                            device=self._device,
                                            top_k_list=(1, 5))
        self._tb_writer.add_scalars(
            f'{type(self._nos.network).__name__} test on '
            f'{type(self._test_dataloader.dataset).__name__}', tag_scalar_dict,
            self._current_epoch)

    @property
    def hparams(self) -> dict:
        return self._hparams

    @staticmethod
    def _train_network(nos: SimpleNOS, dataloader: DataLoader,
                       device: str) -> dict[str, float]:
        network = nos.network
        optimizer = nos.optimizer

        network.train()
        network.to(device)

        loss_meter = AverageMeter(name='loss')
        top1_meter = AverageMeter(name='top1_acc')
        top5_meter = AverageMeter(name='top5_acc')

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = network(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            loss_meter.update(loss.item(), batch_size)
            top1_acc, top5_acc = calc_batch_acc(logits, y, (1, 5))
            top1_meter.update(top1_acc, batch_size)
            top5_meter.update(top5_acc, batch_size)

        return {
            'loss': loss_meter.avg,
            'top1_acc': top1_meter.avg,
            'top5_acc': top5_meter.avg
        }
