import numpy as np
import torch

class ImageStackNormalizationTransform(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def __call__(self, image_stack):
        if isinstance(image_stack, torch.Tensor):
            self.mean = torch.mean(image_stack)
            self.std = torch.std(image_stack)
            normalized_stack = (image_stack - self.mean) / self.std
            return normalized_stack
        else:
            raise TypeError("Input image_stack should be a torch.Tensor.")


class RandomCrop3D(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (tuple, list))
        self.output_size = output_size

    def __call__(self, sample):
        depth, height, width = sample.shape
        new_depth, new_height, new_width = self.output_size
        assert depth >= new_depth and height >= new_height and width >= new_width, "Crop size exceeds image size"
        d_start = np.random.randint(depth - new_depth + 1)
        h_start = np.random.randint(height - new_height + 1)
        w_start = np.random.randint(width - new_width + 1)
        return sample[d_start:d_start+new_depth, h_start:h_start+new_height, w_start:w_start+new_width]

class Crop3D(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (tuple, list))
        self.output_size = output_size

    def __call__(self, sample):
        #depth, height, width = sample.shape
        depth, height, width = tuple(x / 2 for x in sample.shape)
        depth, height, width = int(depth), int(height), int(width)
        new_depth, new_height, new_width = self.output_size
        assert depth >= new_depth and height >= new_height and width >= new_width, "Crop size exceeds image size"
        d_start = (depth - new_depth) // 2
        h_start = (height - new_height) // 2
        w_start = (width - new_width) // 2
        return sample[d_start:d_start+new_depth, h_start:h_start+new_height, w_start:w_start+new_width]
