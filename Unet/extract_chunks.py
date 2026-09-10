import os
import numpy as np
import tifffile
from pathlib import Path

# define input and output directories
input_dir = '/bil/proj/rf1hillman/results/slab21_pfCortex_chunks/masks/'
output_dir = '//bil/proj/rf1hillman/results/slab21_pfCortex_chunks/masks_chunked/'

# create the output directory if it does not already exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# define chunk size
chunk_size = (128, 128, 128)

# loop over all files in input directory
for filename in os.listdir(input_dir):

    # read in original tiff stack
    original_stack = tifffile.imread(os.path.join(input_dir, filename))

    # calculate number of chunks in each dimension
    num_chunks = [int(np.ceil(dim / chunk_size[i])) for i, dim in enumerate(original_stack.shape)]
    print(f'Number of chunks is {num_chunks}')

    # loop over all chunks
    for z in range(num_chunks[0]):
        for y in range(num_chunks[1]):
            for x in range(num_chunks[2]):
                # calculate chunk indices
                z_start, z_stop = z * chunk_size[0], (z + 1) * chunk_size[0]
                y_start, y_stop = y * chunk_size[1], (y + 1) * chunk_size[1]
                x_start, x_stop = x * chunk_size[2], (x + 1) * chunk_size[2]

                # slice out chunk from original stack
                chunk = original_stack[z_start:z_stop, y_start:y_stop, x_start:x_stop]

                #pad with 0s
                pad_width = tuple(
                    (0, desired - actual)
                    for actual, desired in zip(chunk.shape, chunk_size)
                )

                chunk = np.pad(
                    chunk,
                    pad_width,
                    mode="constant",
                    constant_values=0,
                )

                # create output filename based on original filename and chunk indices
                stem = Path(filename).stem

                output_filename = (
                    f"{stem}"
                    f"_z{z_start:06d}"
                    f"_y{y_start:06d}"
                    f"_x{x_start:06d}.tif"
                )
                #output_filename = f'{filename[:-4]}_{z_start}_{y_start}_{x_start}.tif'

                # write chunk to output directory
                tifffile.imwrite(os.path.join(output_dir, output_filename), chunk)
