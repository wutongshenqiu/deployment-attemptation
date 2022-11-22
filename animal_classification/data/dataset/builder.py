import copy

from torch.utils.data import Dataset
from torchvision import transforms

DATASETS_MAPPING: dict[str, Dataset] = {}


def build_dataset(dataset_cfg: dict) -> Dataset:
    dataset_cfg = copy.deepcopy(dataset_cfg)

    if 'type' not in dataset_cfg:
        raise ValueError('Dataset config must have `type` field')
    dataset_type = dataset_cfg.pop('type')
    if dataset_type not in DATASETS_MAPPING:
        raise ValueError(
            f'Dataset `{dataset_type}` is not support, '
            f'available datasets: {list(DATASETS_MAPPING.keys())}')
    dataset = DATASETS_MAPPING[dataset_type]

    if 'target_transform' in dataset_cfg:
        raise ValueError('`target_transform` is not support')

    if 'transform' in dataset_cfg:
        transform_list = []
        transform_list_cfg = dataset_cfg.pop('transform')
        for transform_cfg in transform_list_cfg:
            transform_type = transform_cfg.pop('type')
            transform = getattr(transforms, transform_type)(**transform_cfg)
            transform_list.append(transform)

        transform = transforms.Compose(transform_list)
    else:
        transform = None

    return dataset(transform=transform, **dataset_cfg)
