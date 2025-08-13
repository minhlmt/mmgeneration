import os.path as osp
import mmcv
from torch.utils.data import Dataset
from .builder import DATASETS
from .pipelines import Compose
import scipy.io as sio
from PIL import Image
import numpy as np

@DATASETS.register_module()
class SVHNDataset(Dataset):
    """SVHN dataset từ file .mat cho Conditional GANs.

    Args:
        mat_file (str): đường dẫn tới file .mat (train_32x32.mat hoặc test_32x32.mat)
        pipeline (list[dict | callable]): danh sách transform
        test_mode (bool): nếu True sẽ load test
    """

    def __init__(self, mat_file, pipeline, test_mode=False):
        super().__init__()
        self.mat_file = mat_file
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode

        self.imgs, self.labels = self.load_annotations()

    def load_annotations(self):
        """Load dữ liệu từ file .mat"""
        data = sio.loadmat(self.mat_file)
        imgs = np.transpose(data['X'], (3, 0, 1, 2))  # (N, H, W, C)
        labels = data['y'].squeeze()  # nhãn từ 1..10, 10 = số 0
        labels[labels == 10] = 0
        return imgs, labels.astype(int)

    def prepare_data(self, idx):
        """Chuẩn bị ảnh và nhãn"""
        img = np.array(self.imgs[idx])  # <-- np.ndarray
        label = int(self.labels[idx])
        results = dict(img=img, gt_label=label)
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def __len__(self):
        return len(self.labels)

    def __repr__(self):
        return f"SVHNDataset(num_images={len(self)}, mat_file='{self.mat_file}')"
