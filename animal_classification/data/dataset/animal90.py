from pathlib import Path

from torchvision.datasets import ImageFolder


class Animal90(ImageFolder):
    split2dir = {'train': 'train', 'val': 'val'}

    def __init__(self, root: str | Path, split: str = 'train', **kwargs):
        root = Path(root) / self.split2dir[split]

        super().__init__(root, **kwargs)
