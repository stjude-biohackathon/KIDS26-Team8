import os
from torch.utils.data import Dataset
import tifffile as tiff
from glob import glob
import torchvision.transforms as transforms
from pathlib import Path


class StackDataset(Dataset):

    def __init__(self, source_dir, target_dir, transform):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.transform = transform

        # Get a list of file paths within the source directory
        self.source_files = glob(os.path.join(source_dir, '*'))
        self.target_files = glob(os.path.join(target_dir, '*'))

        if len(self.source_files) == 0 or len(self.target_files) == 0:
            raise ValueError("The source or target directory is empty.")

        self.source_files.sort()
        self.target_files.sort()

        # Ensure the same number of raw images and masks
        if len(self.source_files) != len(self.target_files):
            raise ValueError(
                f"Different numbers of files: "
                f"{len(self.source_files)} raw files and "
                f"{len(self.target_files)} masks."
            )

        # Ensure every raw file is paired with a mask having the same filename
        source_names = [Path(path).name for path in self.source_files]
        target_names = [Path(path).name for path in self.target_files]

        if source_names != target_names:
            source_only = sorted(set(source_names) - set(target_names))
            target_only = sorted(set(target_names) - set(source_names))

            raise ValueError(
                "Raw and mask filenames do not match.\n"
                f"Only in raw directory: {source_only[:10]}\n"
                f"Only in mask directory: {target_only[:10]}"
            )

        print(
            f"Found {len(self.source_files)} matched raw/mask pairs."
        )

    def __len__(self):
        #num_of_imgs = 0
        #for tiff_path in self.source_files:
        #    num_of_imgs += tiff.imread(tiff_path).shape[0]
        #return num_of_imgs
        num_of_stacks = len(self.source_files)
        return num_of_stacks

    def __getitem__(self, index):
        source_path = self.source_files[index]
        target_path = self.target_files[index]
        source = tiff.imread(source_path).astype('float32') #.astype('int8')
        target = tiff.imread(target_path).astype('bool')

        if self.transform:
            source = self.transform(source)
            target = self.transform(target)
            #target = transforms.ToTensor()(target)
        # Convert numpy arrays to PyTorch tensors
        #source = torch.from_numpy(source)
        #target = torch.from_numpy(target)

        # Add a channel dimension to the tensors (assuming grayscale images)
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)
        return source, target
