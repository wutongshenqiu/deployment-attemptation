# TODO: Use proper name
# `nos` means network + optimizer + wrapper

import copy
from typing import Iterable, Optional

from torch import nn, optim
from torch.nn import Module, Parameter

from animal_classification.network import build_network


def _build_module(module_cfg: dict) -> Module:
    if 'type' not in module_cfg:
        raise ValueError('Config must have `type` field')

    module_cfg = copy.deepcopy(module_cfg)

    module_type = module_cfg.pop('type')
    if (module := getattr(nn, module_type)) is None:
        raise ValueError(f'`{module_type}` is not supported!')

    return module(**module_cfg)


def _replace_module(model: Module, mapping_cfg: dict[str, dict]) -> None:
    used_modules = set()

    def traverse_children(module: Module, prefix: str) -> None:
        for name, child in module.named_children():
            if prefix == '':
                child_name = name
            else:
                child_name = f'{prefix}.{name}'
            if child_name in mapping_cfg:
                new_module = _build_module(mapping_cfg[child_name])
                print(f'Module `{child_name}` will be replaced')
                print(f'Original: {child}')
                print(f'New: {new_module}')
                setattr(module, name, new_module)
                used_modules.add(child_name)
            else:
                traverse_children(child, child_name)

    traverse_children(model, '')

    if (unused_modules := set(mapping_cfg.keys()) - used_modules):
        raise ValueError(f'Unexpected modules: {unused_modules}')


class SimpleNOS(Module):
    """One network + one optimizer + one scheduler"""

    def __init__(
            self,
            *,
            network: Module,
            optimizer: optim.Optimizer,
            scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ) -> None:
        super().__init__()

        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler

    @classmethod
    def build_from_cfg(cls, cfg: dict) -> 'SimpleNOS':
        # avoid circular import
        from .builder import build_optimizer, build_scheduler

        if 'network' not in cfg:
            raise ValueError('Config must have `network` field')
        if 'optimizer' not in cfg:
            raise ValueError('Config must have `optimizer` field')

        cfg = copy.deepcopy(cfg)

        network_cfg = cfg.pop('network')
        network: Module = build_network(network_cfg)

        optimizer_cfg = cfg.pop('optimizer')
        optimizer: optim.Optimizer = build_optimizer(
            params=network.parameters(), optimizer_cfg=optimizer_cfg)

        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if 'scheduler' in cfg:
            scheduler_cfg = cfg.pop('scheduler')
            scheduler = build_scheduler(optimizer=optimizer,
                                        scheduler_cfg=scheduler_cfg)

        # HACK: inconsistent with `__init__` signature
        return cls(network=network,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   **cfg)

    def state_dict(self,
                   *args,
                   destination=None,
                   prefix='',
                   keep_vars=False) -> dict:
        state_dict = super().state_dict(*args,
                                        destination=destination,
                                        prefix=prefix,
                                        keep_vars=keep_vars)
        state_dict[f'{prefix}optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict[f'{prefix}scheduler'] = self.scheduler.state_dict()

        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        optimizer_key = f'{prefix}optimizer'
        scheduler_key = f'{prefix}scheduler'
        if optimizer_key in state_dict:
            self.optimizer.load_state_dict(state_dict.pop(optimizer_key))
        if scheduler_key in state_dict:
            self.scheduler.load_state_dict(state_dict.pop(scheduler_key))

        return super()._load_from_state_dict(state_dict, prefix,
                                             local_metadata, strict,
                                             missing_keys, unexpected_keys,
                                             error_msgs)


class SimpleFreezeNOS(SimpleNOS):

    @classmethod
    def build_from_cfg(cls, cfg: dict) -> 'SimpleFreezeNOS':
        # avoid circular import
        from .builder import build_optimizer, build_scheduler

        if 'network' not in cfg:
            raise ValueError('Config must have `network` field')
        if 'optimizer' not in cfg:
            raise ValueError('Config must have `optimizer` field')
        if 'training' not in cfg:
            raise ValueError('Config must have `training` field')

        cfg = copy.deepcopy(cfg)

        network_cfg = cfg.pop('network')
        network: Module = build_network(network_cfg)

        if 'mapping' in cfg:
            mapping_cfg: dict[str, dict] = cfg.pop('mapping')
            _replace_module(network, mapping_cfg)

        training_cfg = cfg.pop('training')
        mode: bool = training_cfg.get('mode', False)
        network.train(mode)

        # freeze all parameters
        for p in network.parameters():
            p.requires_grad = False

        # collect trainable parameters
        name2trainable_paramter: dict[str, Parameter] = \
            cls.collect_trainable_parameters(
                network, training_cfg['trainable_modules'])
        for name, parameter in network.named_parameters():
            if name in name2trainable_paramter:
                parameter.requires_grad = True

        print('Trainable parameters')
        for name, parameter in name2trainable_paramter.items():
            print(f'name: `{name}`, shape: {parameter.shape}')

        optimizer_cfg = cfg.pop('optimizer')
        optimizer: optim.Optimizer = build_optimizer(
            params=name2trainable_paramter.values(),
            optimizer_cfg=optimizer_cfg)

        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if 'scheduler' in cfg:
            scheduler_cfg = cfg.pop('scheduler')
            scheduler = build_scheduler(optimizer=optimizer,
                                        scheduler_cfg=scheduler_cfg)

        # HACK: inconsistent with `__init__` signature
        return cls(network=network,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   **cfg)

    @staticmethod
    def collect_trainable_parameters(
            model: Module,
            trainable_module_names: Iterable[str]) -> dict[str, Parameter]:
        name2trainable_paramter: dict[str, Parameter] = dict()

        trainable_module_name_set = set(trainable_module_names)
        used_module_set = set()
        for module_name, module in model.named_modules():
            if module_name in trainable_module_name_set:
                for parameter_name, parameter in module.named_parameters():
                    name2trainable_paramter[
                        f'{module_name}.{parameter_name}'] = parameter

                used_module_set.add(module_name)

        if used_module_set != trainable_module_name_set:
            print('Unexpected module: '
                  f'{trainable_module_name_set - used_module_set}')

        return name2trainable_paramter
