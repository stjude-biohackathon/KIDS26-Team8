# Biohackathon Project Template

This repository is a starting point for a three-day team project. This repository is populated with a starting template for team organization and planning. Use it to plan, build, and document work. Please adjust this repository to suit the needs of your team.

> **Team leads:** Start with the [team lead checklist](project-management/CHECKLIST.md) before the event or during your first team meeting.

## Project Profile

- **Project name:** [Add a short, descriptive name]
- **Question, problem, or opportunity:** [What are you exploring?]
- **Data, inputs, or evidence:** [What will you use, and where does it come from?]
- **Expected output:** [What will you show, test, explain, or demonstrate?]
- **Tools and stack:** [Languages, libraries, notebooks, APIs, databases, services, or other tools]
- **Team lead:** [Name and GitHub handle]
- **Team members and roles:** [Link to `project-management/team.md`]
- **Communication:** [Add the agreed channel or contact]

Naming the tools and stack early helps the team lead create useful roles and divide work realistically. It is fine to revise this section as the project develops.

## Vision and Mission

- **Vision:** [Describe the change, insight, or capability you hope this project supports.]
- **Mission:** [Describe what the team will do during the biohackathon to move toward that vision.]

## About

[Add a short explanation of the motivation, background, and why the question or problem matters.]

## Roadmap and Milestones

| When | Focus | Expected outcome |
| --- | --- | --- |
| Day 1 | Agree on the question, inputs, stack, roles, and first tasks | A shared plan and a first small change in the repository |
| Day 2 | Build, test, and compare approaches | A working result or clear evidence about what does not work |
| Day 3 | Stabilize, document, and present | A demo or handoff with methods, limitations, and next steps |

The goal is not a perfect production system. The goal is a clear, honest, useful result that the team can explain and others can build on.

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


