from pathlib import Path
import os
from torch.utils.data import Dataset
from Register import Registers
from datasets.utils import get_image_paths_from_dir
from datasets.base import ImagePathDataset

# Task에 맞게 데이터 구성
@Registers.datasets.register_with_name('custom_ct_translation')
class CustomCTTranslationDataset(Dataset):
    """
    NECT -> CECT CT image translation.
    충족되어야 할 directory structure:

    dataset_path/
        train/
            NECT/
                S0001_NECT_0001.png
                S0002_NECT_0001.png
                ...
            CECT/
                S0001_CECT_0001.png
                S0002_CECT_0001.png
                ...
        val/
            NECT/
            CECT/
        test/
            NECT/
            CECT/
    """

    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.dataset_root = Path(dataset_config.dataset_path)
        self.nect_paths = sorted(get_image_paths_from_dir(self.dataset_root / stage / "NECT"))  # NECT
        self.cect_paths = sorted(get_image_paths_from_dir(self.dataset_root / stage / "CECT"))  # CECT

        assert len(self.nect_paths) == len(self.cect_paths), "NECT and CECT slice counts must match."

        self.nect_dataset = ImagePathDataset(self.nect_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.cect_dataset = ImagePathDataset(self.cect_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.nect_dataset)

    def __getitem__(self, idx):
        target_img, name = self.cect_dataset[idx]  # Target: CECT, shape (C, H, W)
        cond_img, _ = self.nect_dataset[idx]       # Condition: NECT, shape (C, H, W)
        return (target_img, name), (cond_img, name)
