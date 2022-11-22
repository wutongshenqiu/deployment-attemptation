import copy
from typing import Iterable, Type

from torch import Tensor, optim

from .nos import SimpleFreezeNOS, SimpleNOS

_NOS_MAPPING = {'SimpleNOS': SimpleNOS, 'SimpleFreezeNOS': SimpleFreezeNOS}


def build_optimizer(params: Iterable[Tensor],
                    optimizer_cfg: dict) -> optim.Optimizer:
    if 'type' not in optimizer_cfg:
        raise ValueError('Optimizer config must have `type` field')
    optimizer_cfg = copy.deepcopy(optimizer_cfg)

    optimizer_type = optimizer_cfg.pop('type')

    if (optimizer := getattr(optim, optimizer_type)) is None:
        raise ValueError(f'Optimizer `{optimizer_type}` is not support')

    return optimizer(params, **optimizer_cfg)


def build_scheduler(optimizer: optim.Optimizer,
                    scheduler_cfg: dict) -> optim.lr_scheduler._LRScheduler:
    if 'type' not in scheduler_cfg:
        raise ValueError('Scheduler config must have `type` field')
    scheduler_cfg = copy.deepcopy(scheduler_cfg)

    scheduler_type = scheduler_cfg.pop('type')

    if (scheduler := getattr(optim.lr_scheduler, scheduler_type)) is None:
        raise ValueError(f'Scheduler `{scheduler_type}` is not support')

    return scheduler(optimizer, **scheduler_cfg)


def build_nos(nos_cfg: dict) -> SimpleNOS:
    nos_cfg = copy.deepcopy(nos_cfg)

    if 'type' not in nos_cfg:
        raise ValueError('NOS config must have `type` field')
    nos_type = nos_cfg.pop('type')
    if nos_type not in _NOS_MAPPING:
        raise ValueError(f'NOS `{nos_type}` is not support, '
                         f'available NOS: {list(_NOS_MAPPING.keys())}')
    nos: Type[SimpleNOS] = _NOS_MAPPING[nos_type]

    # HACK
    return nos.build_from_cfg(nos_cfg)
