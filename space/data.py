import pathlib
from typing import Optional, Dict, Tuple, List

import numpy as np

import torch

import h5py

TEST = '''
file1001126.h5
file1001930.h5
file1002436.h5
file1002382.h5
file1001585.h5
file1001506.h5
file1001057.h5
file1002021.h5
file1000831.h5
file1001344.h5
file1000007.h5
file1001938.h5
file1001381.h5
file1001365.h5
file1001458.h5
file1001726.h5
file1000280.h5
file1002145.h5
file1001759.h5
file1000976.h5
file1001064.h5
file1000108.h5
file1000291.h5
file1001689.h5
file1001191.h5
file1000903.h5
file1001440.h5
file1001184.h5
file1001119.h5
file1002351.h5
file1001643.h5
file1001893.h5
file1001968.h5
file1001566.h5
file1001850.h5
file1000660.h5
file1000593.h5
file1001763.h5
file1002546.h5
file1000697.h5
file1000190.h5
file1000273.h5
file1001144.h5
file1000538.h5
file1000635.h5
file1000769.h5
file1001262.h5
file1001851.h5
file1000942.h5
'''

VAL = '''
file1002340.h5
file1000182.h5
file1001655.h5
file1000972.h5
file1001338.h5
file1000476.h5
file1002252.h5
file1000591.h5
file1000858.h5
file1001202.h5
file1002159.h5
file1001163.h5
file1001497.h5
file1000196.h5
file1001331.h5
file1000477.h5
file1001651.h5
file1000464.h5
file1000625.h5
file1000033.h5
file1000041.h5
file1000000.h5
file1000389.h5
file1001668.h5
file1002451.h5
file1000990.h5
file1001077.h5
file1002389.h5
file1001298.h5
file1002570.h5
file1001834.h5
file1001955.h5
file1001715.h5
file1001444.h5
file1002214.h5
file1002187.h5
file1001170.h5
file1000926.h5
file1000818.h5
file1000702.h5
file1001289.h5
file1000528.h5
file1000759.h5
file1001862.h5
file1001339.h5
file1001090.h5
file1002067.h5
file1001221.h5
'''


class SingleCoilKneeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for handling single-coil knee MRI data from the fastMRI dataset,
    following the work of `Pineda et al. MICCAI'20 <https://arxiv.org/pdf/2007.10469.pdf>`_.
    """
    K_SPACE_WIDTH = 368
    K_SPACE_HEIGHT = 640

    def __init__(
        self,
        root: str,
        mode: str,
        slice_ids: Optional[List[int]] = None,
    ):
        """
        :param root: Root directory containing the dataset files.
        :param mode: Dataset mode ('train', 'val', or 'test').
        :param slice_ids: List of slice indices to include. If None, all slices are include
        """
        files = self._filter_files(root, mode)
        self.metadata = []
        for volume_i, filename in enumerate(sorted(files)):
            data = h5py.File(filename, "r")
            for slice_id in range(data["kspace"].shape[0]):
                if slice_ids is not None:
                    if slice_id not in slice_ids:
                        continue
                self.metadata.append((filename, slice_id))

    def _filter_files(self, root, mode):
        files = []
        for filename in list(pathlib.Path(root).iterdir()):
            data = h5py.File(filename, 'r')
            if data['kspace'].shape[2] != self.K_SPACE_WIDTH:
                continue
            files.append(filename)
        if mode == 'train':
            return files
        assert mode in ('test', 'val'), f'Invalid mode: {mode}.'
        filter_set = set((TEST if mode == 'test' else VAL).strip().splitlines())
        return [f for f in files if f.name in filter_set]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, Dict, str, int]:
        filename, slice_num = self.metadata[index]
        with h5py.File(filename, "r") as data:
            k_space = data["kspace"][slice_num]
            k_space = torch.from_numpy(k_space.view(np.float32).reshape(self.K_SPACE_HEIGHT, self.K_SPACE_WIDTH, 2))
            k_space = torch.view_as_complex(k_space)
            k_space = torch.fft.ifftshift(k_space, dim=(0, 1))
            image = torch.fft.ifft2(k_space, norm='backward')
            image = torch.fft.fftshift(image, dim=(0, 1))
            # Precompute the mean norm from the training data.
            image /= 7.072103529760345e-07
            k_space /= 7.072103529760345e-07
            image, k_space = torch.view_as_real(image), torch.view_as_real(k_space)  # (H, W, 2)
            return image, k_space, dict(data.attrs), filename.name, slice_num
