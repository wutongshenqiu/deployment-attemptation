import dataclasses
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.nn import Module
from torchvision.datasets.folder import default_loader
from torchvision.transforms._presets import ImageClassification

from animal_classification.component import build_nos
from animal_classification.config import Config

_DEFAULT_TRANSFORM = ImageClassification(crop_size=224,
                                         resize_size=256,
                                         mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))

_PATH_TYPE = str | Path


@dataclasses.dataclass
class ClassifierResult:
    probabilities: list[float]
    predictions: list[int]


def init_classifier(config: str | Config,
                    device: str = 'cuda'
                    if torch.cuda.is_available() else 'cpu',
                    checkpoint: Optional[_PATH_TYPE] = None) -> Module:
    if isinstance(config, str):
        config = Config.fromfile(config)
    if not isinstance(config, Config):
        raise TypeError('`config` must be a filename or Config object, '
                        f'but got: {config}')

    nos_cfg = config.trainer.nos
    nos = build_nos(nos_cfg)
    classifier = nos.network

    if checkpoint is not None:
        classifier.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    classifier.to(device)
    classifier.eval()

    return classifier


@torch.no_grad()
def inference_classifier(classifier: Module,
                         imgs: _PATH_TYPE | list[_PATH_TYPE]
                         | tuple[_PATH_TYPE],
                         transform: Optional[Callable] = _DEFAULT_TRANSFORM,
                         loader: Callable[[_PATH_TYPE], Any] = default_loader):
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    imgs = [Path(img) for img in imgs]

    for img in imgs:
        # HACK: just to meet mypy
        assert isinstance(img, Path)
        if not img.exists():
            raise ValueError(f'Image file `{img}` does not exist')

    device = next(classifier.parameters()).device

    img_list = [loader(x) for x in imgs]
    if transform is not None:
        img_list = [transform(img) for img in img_list]
    batch_x = torch.stack(img_list, dim=0)
    print(batch_x.shape)
    batch_x = batch_x.to(device)

    logits: torch.Tensor = classifier(batch_x)
    probabilities = logits.softmax(dim=1)
    predictions = torch.argmax(probabilities, dim=1)

    return ClassifierResult(probabilities=[
        probabilities[i, predictions[i]] for i in range(batch_x.shape[0])
    ],
                            predictions=predictions.tolist())
