import torch
import torchvision
import os
from timeit import default_timer as timer
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
from unet_class import UNet3D
import torch.optim as optim
import torchvision.transforms as transforms
import train as tr
import transforms as t
import datasets_class as dt
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 4
NUM_EPOCHS = 1
VAL_SPLIT = 0.2
NUM_LAYERS = 3
CLASS_WEIGHT = 300  # this should be the ration of negative pixels / positive pixels (i.e. background/mask)

# Specify the paths to the source and target image stack directories
source_dir =  '/bil/proj/rf1hillman/results/slab21_pfCortex_chunks/raw_chunked/' 
target_dir =  '/bil/proj/rf1hillman/results/slab21_pfCortex_chunks/masks_chunked/' 
model_dir = f'/bil/users/psimko/holis/stJude_hackathon/Unet/models/'
model_path = os.path.join(model_dir, "model"+".pth")
model_name = f'hackathonStJude_pfCortex_Layers{NUM_LAYERS}_LR{LEARNING_RATE}_BS{BATCH_SIZE}_Ep{NUM_EPOCHS}_Vs{VAL_SPLIT}_CLASS_W_{CLASS_WEIGHT}'

# Create the output directory if it does not already exist
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Set CUDA_LAUNCH_BLOCKING to 1
#torch.backends.cudnn.benchmark = False # Disable cudnn.benchmark

# Define the transforms to apply to the images
train_transform = transforms.Compose([
    # You can add additional transforms here, such as resizing, rotation, etc.
    transforms.ToTensor(),
    #t.ImageStackNormalizationTransform(),
    #t.RandomCrop3D(output_size=(128, 128, 128)),  # adjust crop size as needed
    #t.Crop3D(output_size=(128, 128, 128))
])

# Initialize a writer for tensorboard visualization
writer = SummaryWriter(log_dir=f'/bil/users/psimko/holis/stJude_hackathon/Unet/runs/metric_tests/{model_name}/')

#=====================================================================================================#

if __name__ == '__main__':
    # Create a dataset (containing both source and target images)
    dataset = dt.StackDataset(source_dir, target_dir, transform=train_transform)
    print(f'Dataset size: {len(dataset)}')

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define the lengths of training and validation sets
    val_size = int(VAL_SPLIT * len(dataset))  # val_split % for validation
    train_size = len(dataset) - val_size      # Remaining (1 - val_split) % for training

    # Use random_split to create training and validation splits and set batch_size
    split_generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)

    # Create dataloaders for datasets for both training and validation
    if len(train_dataset) == 0:
        print("Train dataset is empty!")
    elif len(train_dataset) < BATCH_SIZE:
        print("Train dataset has fewer elements than batch size!")
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
            )

    if len(val_dataset) == 0:
        print("Validation dataset is empty!")
    elif len(val_dataset) < BATCH_SIZE:
        print("Validation dataset has fewer elements than batch size!")
    else:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
            )

    # Before CNN definition, check the sizing of input tensor
    data, label = next(iter(train_dataloader))
    print(data.size())
    print(label.size())
    data, label = next(iter(val_dataloader))
    print(data.size())
    print(label.size())

    # Define model
    model = UNet3D(num_layers=NUM_LAYERS).to(device)
    print(model)

    # Define class weights
    class_weights = torch.tensor([CLASS_WEIGHT], dtype=torch.float32, device=device)         # this should be the ratio of negative pixels / positive pixels (i.e. background/mask)

    # Specify a loss function and an optimizer to be used for training
    loss_fn = nn.BCEWithLogitsLoss(pos_weight = class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Create checkpoint directories
    checkpoint_dir = os.path.join(
        model_dir,
        model_name,
        "checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_model_path = os.path.join(
        model_dir,
        f"{model_name}_best.pth"
    )

    latest_model_path = os.path.join(
        model_dir,
        f"{model_name}_latest.pth"
    )

    # Train the model
    epoch_number = 0
    train_accuracies_per_epoch = []  # cumulated accuracies from training dataset for each epoch
    val_accuracies_per_epoch = []  # cumulated accuracies from validation dataset for each epoch

    config = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "val_split": VAL_SPLIT,
        "num_layers": NUM_LAYERS,
        "class_weight": CLASS_WEIGHT,
    }

    best_val = -float("inf")

    start = timer()
    for epoch in range(NUM_EPOCHS):
        train_accuracy_epoch, val_accuracy_epoch, train_jaccard_epoch, val_jaccard_epoch, train_dice_epoch, val_dice_epoch = tr.train_one_epoch(epoch, NUM_EPOCHS, train_dataloader, val_dataloader, model, loss_fn, optimizer, device)

        writer.add_scalar("Nuclei pixel accuracy/train", train_accuracy_epoch, epoch)
        writer.add_scalar("Nuclei pixel accuracy/val", val_accuracy_epoch, epoch)

        writer.add_scalar("Dice score/train", train_dice_epoch, epoch)
        writer.add_scalar("Dice score/val", val_dice_epoch, epoch)

        writer.add_scalar("Jaccard score/train", train_jaccard_epoch, epoch)
        writer.add_scalar("Jaccard score/val", val_jaccard_epoch, epoch)

        torch.save(model.state_dict(), latest_model_path)

        is_best = val_dice_epoch > best_val

        if is_best:
            best_val = val_dice_epoch
            torch.save(model.state_dict(), best_model_path)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_val_dice": best_val,
            "config": config,
        }

        # save latest checkpoint
        torch.save(ckpt, os.path.join(checkpoint_dir, "latest.ckpt"))

        # save best
        if is_best:
            torch.save(ckpt, os.path.join(checkpoint_dir, "best.ckpt"))


    writer.flush()
    writer.close()
    print("Done!")
