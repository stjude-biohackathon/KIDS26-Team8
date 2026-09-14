# Counting every cell in the human brain: 3D localization of cell nuclei

## Project Profile

- **Name:** Counting every cell in the human brain: 3D localization of cell nuclei
- **Question:** Our goal is to accurately detect and localize nuclei while minimizing computational cost, enabling analysis at whole-human-brain scale (>150 billion cells).
- **Data:** Participants will have access to a subset of volumetric microscopy data acquired as part
of the Human Brain Optimized Light-sheet (HOLiS) NIH project (total dataset > 6 PB!).

  The dataset was collected on 5 mm thick optically cleared coronal human hemibrain slabs stained with a nuclear dye and 4 immunohistochemistry markers and imaged using a custom oblique-plane single-objective light-sheet microscopy system. The main data is several 3D strips of nuclear channel data from three different brain regions representing expected variations in cellular density. Additional data includes four auxiliary imaging channels for antibody-based labeling (e.g., vasculature markers).

   The volumetric microscopy data is provided by the Hillman Lab who designed and
   built the HOLiS microscope and performed the imaging.

   Immunostaining was performed by the Wu Lab at the Weill Cornell Medicine Helen &
   Robert Appel Alzheimer’s Disease Research Institute, with brain tissue provided by Dr
   John Crary, Director of Mt Sinai’s Neuropathology Brain Bank.
   Data is used for segmentation validation but will not be publicly posted at the
   conclusion of the BioHackathon.
- **Expected output:** Additional nuclear segmentation pipeline options, balancing segmentation or expression alignment accuracy with runtime and memory efficiency. Benchmarked against our baseline approaches (binarization and 3D U-Net), including lightweight quality-control or confidence metrics. Pipeline options should be scalable and suitable for integration into large-scale analysis workflows.
- **Tools and stack:**  Helpful analysis and visualization tools include, but are not limited to: Python, PyTorch, Slurm, Neuroglancer, Ilastik, and existing segmentation methods. 
- **Team leads:** Caitlin Freeman ([@caitlinfree](https://github.com/caitlinfree)) and Peter Simko ([@ps3348](https://github.com/ps3348))
- **Team members and roles:** [project-management/team.md](https://github.com/stjude-biohackathon/KIDS26-Team8/blob/main/project-management/team.md)
- **Communication:** https://app.slack.com/client/T04JD4M0H29/C0BT8BBGWP2


## Vision and Mission

- **Vision:** Enable scalable, accurate analysis of the human brain at cellular resolution, providing new insight into how cellular organization changes across individuals, development, and neurological and psychiatric disease.
- **Mission:** Develop and benchmark efficient, accurate methods for 3D nuclear segmentation and localization in large-scale volumetric microscopy data. By improving segmentation speed, accuracy, quality control, and pipeline throughput, we aim to make analysis of massive whole-brain datasets more accessible and enable quantitative comparisons of cellular organization across human brains.

## About

The data used for this challenge were generated as part of an NIH BRAIN Initiative project focused on building the first dataset surveying the human brain at cellular resolution. This resource provides an unprecedented view of the cellular organization of the human brain, creating opportunities to better understand the biological basis of brain function in both health and disease.

Accurate and scalable nuclear segmentation and localization are essential first steps in analyzing these datasets. Because the data exceed 6 PB and contain billions of cells across enormous volumetric scales, they must be reduced into simpler, accessible representations that enable quantitative analysis of cell-type distributions. Even modest improvements in segmentation speed or accuracy could dramatically accelerate downstream analysis and make it feasible to analyze multiple human brains.

Large-scale comparisons across individuals, developmental stages, and disease states are essential for understanding how neurological and psychiatric diseases alter brain anatomy and cellular organization. The resulting insights could help identify potential therapeutic targets and inform future clinical interventions. Additionally, methods developed through this challenge could extend beyond neuroscience to other large-scale volumetric imaging applications, including different microscopy modalities, biological tissues, and biomedical imaging workflows.

## Roadmap and Milestones

| When | Focus | Expected outcome |
| --- | --- | --- |
| Day 1 | Agree on the question, inputs, stack, roles, and first tasks | A shared plan and a first small change in the repository |
| Day 2 | Build, test, and compare approaches | A working result or clear evidence about what does not work |
| Day 3 | Stabilize, document, and present | A demo or handoff with methods, limitations, and next steps |


## Sample Segmentation Pipeline

### Ilastik Annotation

The goal of this step is to create a probability mask for each training volume using ilastik.

1. **Create a Pixel Classification project**
   - Open ilastik and create a new **Pixel Classification** project.

2. **Add input data**
   - Add multiple `.tif` volumes as **separate images** within the same project.
   - Do **not** load the volumes as a sequence.

3. **Select features**
   - Choose feature scales based on the approximate size of the nuclei.
   - For example, for nuclei approximately 15 pixels in diameter, the ideal scale is approximately:
     
     `σ = 15 / 2 = 7.5`
     
   - Select several scales around this value. For example:
     
     `σ = 1.6, 3.5, 10`

4. **Annotate the images**
   - Annotate the volumes sparsely.
   - A few representative pixels for each class are sufficient; it is not necessary to manually segment entire nuclei.

5. **Configure export settings**
   - Export the probability maps as **HDF5 (`.h5`)** files.
   - Use the following settings:
     - **Data type:** `float32`
     - **Axis order:** `zyxc`
     - **Dataset name:** `exported_data`
     - **Output filename:**
       
     `{dataset_dir}/probabilities/{nickname}_{result_type}.h5`

6. **Export probability maps**
   - Select **Export All**.
   - ilastik will create one `.h5` probability file for each input volume.

---

### Prepare Training Data

The goal of this step is to format the training data for the U-Net.

#### 1. Binarize the Probability Maps

Convert each probability map produced by ilastik into a binary segmentation mask by thresholding.

- Run `proba_to_binary.py` on each probability map.
- Save the resulting binary masks in a `masks/` directory.

For example:

```text
data/
├── raw/
├── probabilities/
└── masks/
```

#### 2. Chunk Data (Raw Volumes and Masks)

Split both the raw image volumes and their corresponding binary masks into smaller chunks for U-Net training.

Chunking is important for two main reasons:

- **GPU memory:** Each training batch must fit into GPU memory. Smaller 3D volumes reduce the memory required during training.
- **U-Net pooling compatibility:** The spatial dimensions of each chunk should be compatible with the pooling operations in the U-Net.

In general, each spatial dimension should be divisible by `2^N`, where `N` is the number of pooling layers.

For example:

- 3 pooling layers → dimensions divisible by `8`
- 4 pooling layers → dimensions divisible by `16`

In the sample pipeline, the chunk size used is:

```text
128 × 128 × 128
```

Run `extract_chunks.py` on each raw volume and its corresponding binary mask.

The chunked data can be saved in:

```text
raw_chunked/
masks_chunked/
```

A typical data organization is:

```text
data/
├── raw/
├── probabilities/
├── masks/
├── raw_chunked/
└── masks_chunked/
```

Each raw chunk should have a corresponding mask chunk with the same spatial dimensions.

---

### Train the U-Net

To train the U-Net, run:

```bash
python main.py
```

Run the training script in the appropriate PyTorch environment on a GPU node.

The output of the training process is a trained model.

#### Hyperparameters

Before starting training, set the main hyperparameters in `main.py`:

- **Learning Rate**  
  Controls the size of each optimization step.

- **Batch Size**  
  Number of training volumes processed together in each batch.  
  The batch size must be chosen so that the full batch fits into GPU memory.

- **Number of Epochs**  
  Number of complete passes through the training dataset.

- **Validation Split**  
  Fraction of the dataset reserved for validation.

- **Number of U-Net Layers**  
  Choose either a **3-layer** or **4-layer** U-Net.

  - 3 layers → input dimensions should generally be divisible by `8`
  - 4 layers → input dimensions should generally be divisible by `16`

- **Class Weight**  
  Used to account for the imbalance between background and nuclei pixels.

  A reasonable starting value is:

  ```text
  class weight = number of negative pixels / number of positive pixels
  ```

  where:

  - **negative pixels** = background pixels
  - **positive pixels** = nuclei pixels

---

### Prediction

After training, use the trained U-Net model to predict nuclei on a new input volume.

Run:

```bash
python predict_volume.py \
    --model path/to/model \
    --input path/to/input_chunk \
    --out-dir path/to/output_dir
```

The prediction script outputs:

- **Binary segmentation mask** (`mask.tif`)
- **Nuclei coordinates** (`centroids.csv`)

Both outputs are written to the directory specified by `--out-dir`.

For example:

```text
out/
├── mask.tif
└── centroids.csv
```

The binary mask contains the predicted nuclei segmentation, while `centroids.csv` contains the coordinates of the detected nuclei.

---

### Possible Ideas to Pursue

#### 1. Detect and Split Nuclei Blobs

We have previously experimented with **dynamic thresholding** to separate nuclei that are incorrectly merged into large blobs.

Possible approach:

1. Run prediction using an initial segmentation threshold.
2. Detect connected components in the resulting mask with unusually large volumes.
3. Re-run segmentation on those regions using a stricter threshold.
4. If the large connected component splits into multiple smaller components, keep the segmentation produced using the stricter threshold.

#### 2. Preprocess Data

Investigate whether additional preprocessing improves segmentation performance.

Possible preprocessing steps include:

- **Scaling**
- **Normalization**
- **Pattern correction**, for example using `BaSiCPy`

These preprocessing steps require additional computation and processing time. An important question is whether the resulting improvement in segmentation quality justifies the additional computational cost.

#### 3. Hyperparameter Tuning

Systematically evaluate different training hyperparameters to determine whether segmentation performance can be improved.

Potential parameters to tune include:

- Learning rate
- Batch size
- Number of epochs
- Number of U-Net layers
- Class weight

#### 4. Validation

Training currently evaluates segmentation primarily using **pixel-wise accuracy metrics**.

Investigate whether validation can be extended to include **object-wise (nuclei-wise) metrics** without substantially increasing training time.

Object-wise validation could help evaluate whether individual nuclei are correctly detected and separated, rather than only measuring agreement at the pixel level.



