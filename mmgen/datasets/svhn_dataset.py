import os.path as osp
import mmcv
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose
import scipy.io as sio
from PIL import Image

@DATASETS.register_module()
class SVHNDataset(Dataset):
    """SVHN dataset for Conditional GANs.

    Args:
        data_prefix (str): folder chứa ảnh train/val/test
        ann_file (str): file nhãn dạng txt (image_name label)
        pipeline (list[dict | callable]): danh sách transform
        test_mode (bool): nếu True sẽ load test
    """

    _VALID_IMG_SUFFIX = ('.jpg', '.png', '.jpeg')

    def __init__(self, data_prefix, ann_file, pipeline, test_mode=False):
        super().__init__()
        self.data_prefix = data_prefix
        self.ann_file = ann_file
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.img_infos = self.load_annotations()

    def load_annotations(self):
        """Load image paths và nhãn từ file txt"""
        img_infos = []
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            img_name, label = line.strip().split()
            img_path = osp.join(self.data_prefix, img_name)
            img_infos.append(dict(img_path=img_path, gt_label=int(label)))
        return img_infos

    def prepare_data(self, idx):
        """Chuẩn bị ảnh và nhãn"""
        results = dict(img=self.img_infos[idx]['img_path'],
                       gt_label=self.img_infos[idx]['gt_label'])
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.img_infos)

    def __repr__(self):
        return f"SVHNDataset(num_images={len(self)}, data_prefix='{self.data_prefix}')"
