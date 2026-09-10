import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from timeit import default_timer as timer

#############################################
from skimage.metrics import contingency_table
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from tifffile import imwrite, imread
from scipy.ndimage import label
from skimage.measure import regionprops
# from skimage import io
from scipy.spatial.distance import cdist
#########################################

#writer = SummaryWriter(log_dir=f'/bil/users/psimko/holis/pynet/human_test_stats/human_data_for_training/runs/metric_tests/{model_name}/')

def train_one_epoch(epoch, num_epochs, train_dataloader, val_dataloader, model, loss_fn, optimizer, device):
    model.train()
    train_accuracy = []
    train_dice = []
    train_jaccard = []
    #predicted_nuclei = []
    #predicted_nuclei_true = [] 
    #loop = tqdm(train_dataloader)
    train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False, mininterval=2.0)

    PROFILE_BATCHES = 20
    previous_batch_end = timer()

    for batch, (x, y) in enumerate(train_bar):

        # Measure how long we waited for DataLoader
        data_time = timer() - previous_batch_end

        # Measure permutation and GPU transfer
        transfer_start = timer()

        x = x.permute(0, 1, 3, 2, 4).contiguous()
        y = y.permute(0, 1, 3, 2, 4).contiguous()
        x, y = x.to(device, dtype=torch.float32, non_blocking=True), y.to(device, dtype=torch.float32, non_blocking=True) #dtype=torch.int8

        transfer_time = timer() - transfer_start

        if epoch == 0 and batch == 0:
            tqdm.write(f"x={x.shape}")
            tqdm.write(f"y={y.shape}")

        # Measure forward and backward
        compute_start = timer()

        # Compute prediction error
        yhat = model(x)
        #print(f'yhat={yhat.shape}')
        #y = F.interpolate(y, size=(64, 64, 64), mode='trilinear', align_corners=False)
        #yhat = F.interpolate(yhat, size=(64, 64, 64), mode='trilinear', align_corners=False)
        loss = loss_fn(yhat, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #prediction = yhat.int()
        #threshold = 0.5
        #prediction = (yhat >= threshold).int()

        compute_time = timer() - compute_start

        ############################################################

        # prediction = (torch.sigmoid(yhat) > 0.5)
        # nuclei_mask = (y > 0)
        # train_accuracy.extend((y[nuclei_mask] == prediction[nuclei_mask]).detach().cpu().numpy())

        # # Jaccard (IoU) and Dice score calculation
        # train_jaccard.append(jaccard_score((y[nuclei_mask], prediction[nuclei_mask]).detach().cpu().numpy()))
        # train_dice.append(f1_score((y[nuclei_mask], prediction[nuclei_mask]).detach().cpu().numpy()))

        ############################################################

        # Measure metrics
        metric_start = timer()

        # Predictions and metrics
        # Convert logits to probabilities and binary predictions
        probability = torch.sigmoid(yhat)
        prediction_tensor = probability > 0.5
        target_tensor = y > 0.5

        # Keep the original 5D shape for object-level measurements below
        prediction_notFlat = prediction_tensor.detach().cpu().numpy()

        # Flatten the complete masks for voxel-level metrics
        prediction = prediction_notFlat.flatten()
        y_true = target_tensor.detach().cpu().numpy().flatten()

        # Whole-mask voxel accuracy
        train_accuracy.append(
            np.mean(prediction == y_true)
        )

        # Whole-mask Jaccard and Dice
        train_jaccard.append(
            jaccard_score(
                y_true,
                prediction,
                zero_division=0
            )
        )

        train_dice.append(
            f1_score(
                y_true,
                prediction,
                zero_division=0
            )
        )

        metric_time = timer() - metric_start

        if epoch == 0 and batch < PROFILE_BATCHES:
            tqdm.write(
                f"Batch {batch:03d} | "
                f"data: {data_time:.3f}s | "
                f"transfer: {transfer_time:.3f}s | "
                f"compute: {compute_time:.3f}s | "
                f"metrics: {metric_time:.3f}s"
            )

        ############################################################

        # mask_ground = y.detach().cpu().numpy()
        # mask_labeled = prediction_notFlat

        # percentage_detected, percentage_detected_true = compare_masks(
        #     mask_ground,
        #     mask_labeled,
        # )

        # if np.isfinite(percentage_detected):
        #     predicted_nuclei.append(percentage_detected)

        # if np.isfinite(percentage_detected_true):
        #     predicted_nuclei_true.append(
        #         percentage_detected_true
        #     )


        ############################################################

        train_bar.set_postfix({'train_acc': np.mean(train_accuracy)})
        #train_bar.update()

    train_accuracy_epoch = np.mean(train_accuracy)*100
    train_jaccard_epoch = np.mean(train_jaccard) * 100
    train_dice_epoch = np.mean(train_dice) * 100

    # predicted_nuclei_epoch = (
    #     np.mean(predicted_nuclei) * 100
    #     if predicted_nuclei
    #     else np.nan
    # )

    # predicted_nuclei_true_epoch = (
    #     np.mean(predicted_nuclei_true) * 100
    #     if predicted_nuclei_true
    #     else np.nan
    # )

    model.eval()
    with torch.no_grad():
        val_accuracy = []
        val_jaccard = []
        val_dice = []
        #val_predicted_nuclei = []
        #val_predicted_nuclei_true = [] 
        val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for vx, vy in val_bar:
            vx = vx.permute(0, 1, 3, 2, 4)
            vy = vy.permute(0, 1, 3, 2, 4)
            vx, vy = vx.to(device, dtype=torch.float32), vy.to(device, dtype=torch.float32)
            yhat = model(vx)
            #prediction = (yhat >= 0.5).int()

            ############################################################

            # prediction = (torch.sigmoid(yhat) > 0.5)
            # nuclei_mask_val = (vy > 0)
            # val_accuracy.extend((vy[nuclei_mask_val] == prediction[nuclei_mask_val]).detach().cpu().numpy())

            # # Jaccard (IoU) and Dice score calculation
            # val_jaccard.append(jaccard_score((vy[nuclei_mask_val], prediction[nuclei_mask_val]).detach().cpu().numpy()))
            # val_dice.append(f1_score((vy[nuclei_mask_val], prediction[nuclei_mask_val]).detach().cpu().numpy()))
            
            ############################################################

            # Convert logits to probabilities and binary predictions
            probability = torch.sigmoid(yhat)
            prediction_tensor = probability > 0.5
            target_tensor = vy > 0.5

            # Keep the original 5D shape for object-level measurements below
            prediction_notFlat = prediction_tensor.detach().cpu().numpy()

            # Flatten the complete masks for voxel-level metrics
            prediction = prediction_notFlat.flatten()
            vy_true = target_tensor.detach().cpu().numpy().flatten()

            # Whole-mask voxel accuracy
            val_accuracy.append(
                np.mean(prediction == vy_true)
            )

            # Whole-mask Jaccard and Dice
            val_jaccard.append(
                jaccard_score(
                    vy_true,
                    prediction,
                    zero_division=0
                )
            )

            val_dice.append(
                f1_score(
                    vy_true,
                    prediction,
                    zero_division=0
                )
            )

            ############################################################

            # mask_ground = vy.detach().cpu().numpy()
            # mask_labeled = prediction_notFlat

            # val_percentage_detected, val_percentage_detected_true = (
            #     compare_masks(
            #         mask_ground,
            #         mask_labeled,
            #     )
            # )

            # if np.isfinite(val_percentage_detected):
            #     val_predicted_nuclei.append(
            #         val_percentage_detected
            #     )

            # if np.isfinite(val_percentage_detected_true):
            #     val_predicted_nuclei_true.append(
            #         val_percentage_detected_true
            #     )


            ############################################################

            val_bar.set_postfix({'val_acc': np.mean(val_accuracy)})
        val_accuracy_epoch = np.mean(val_accuracy)*100
        val_jaccard_epoch = np.mean(val_jaccard) * 100
        val_dice_epoch = np.mean(val_dice) * 100

        # val_predicted_nuclei_epoch = (
        #     np.mean(val_predicted_nuclei) * 100
        #     if val_predicted_nuclei
        #     else np.nan
        # )

        # val_predicted_nuclei_true_epoch = (
        #     np.mean(val_predicted_nuclei_true) * 100
        #     if val_predicted_nuclei_true
        #     else np.nan
        # )


    print(f'Epoch #{epoch + 1}. Train accuracy: {train_accuracy_epoch:.2f}. \
                                Validation accuracy: {val_accuracy_epoch:.2f}')
    #loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
    #loop.set_postfix(loss= loss, acc=train_accuracy_epoch)
    return (
        train_accuracy_epoch,
        val_accuracy_epoch,
        train_jaccard_epoch,
        val_jaccard_epoch,
        train_dice_epoch,
        val_dice_epoch,
    )

def calculate_stat(mask_ground, mask):
    mask_ground, _ = label(mask_ground)
    mask, _ = label(mask)

    t = contingency_table(mask_ground, mask)

    # properties of ground truth
    df_ground = pd.DataFrame(regionprops_table(mask_ground,properties=['label','area']))
    df_ground['status'] = None

    # properties of detected
    df_detected = pd.DataFrame(regionprops_table(mask,properties=['label','area']))
    df_detected['status'] = None

    ground_objects = np.unique(mask_ground)
    detected_objects = np.unique(mask)

    # find false positive
    for j in detected_objects:

        match_obj = (t[:,j]>0).nonzero()

        if (match_obj[0].shape[0]<2):

            df_detected.loc[df_detected.label==j,'status'] = 'FP'

    # find true positive and true negative
    for i in ground_objects:

        # if there is a match
        if np.max(t[i,:]) > 0:

            candidate_obj = (t[i,:] == np.max(t[i,:])).nonzero()[1][0]

            iou = np.max(t[i,:]) / np.sum(((mask_ground == i) | (mask==candidate_obj)))

            if iou > 0.8:

                df_ground.loc[df_ground.label==i,'status'] = 'TP'


            else:

                df_ground.loc[df_ground.label==i,'status'] = 'FN'


        else:

            df_ground.loc[df_ground.label==i,'status'] = 'FN'

    tp = np.sum(df_ground.status=='TP')
    fn = np.sum(df_ground.status=='FN')
    fp = np.sum(df_detected.status=='FP')

    return tp,fn,fp


def compare_masks(
    ground_truth_mask,
    predicted_mask,
    single_nucleus_max_distance=10.0,
):
    """
    Compare connected components using centroid distance.

    Returns
    -------
    predicted_count_ratio:
        Number of predicted objects / number of ground-truth objects.

    detected_fraction:
        Fraction of ground-truth objects having at least one predicted
        centroid within the matching distance.

    For chunks without ground-truth nuclei, both metrics are undefined
    and np.nan is returned.
    """

    predicted_mask = predicted_mask[0, 0]
    ground_truth_mask = ground_truth_mask[0, 0]

    ground_truth_labels, num_ground_truth = label(
        ground_truth_mask
    )
    predicted_labels, num_predicted = label(
        predicted_mask
    )

    # Detection metrics are undefined when there are no true nuclei.
    if num_ground_truth == 0:
        return np.nan, np.nan

    # No predictions means zero detected nuclei.
    if num_predicted == 0:
        return 0.0, 0.0

    ground_truth_props = regionprops(ground_truth_labels)
    predicted_props = regionprops(predicted_labels)

    ground_truth_centroids = np.asarray(
        [prop.centroid for prop in ground_truth_props],
        dtype=np.float32,
    ).reshape(-1, 3)

    predicted_centroids = np.asarray(
        [prop.centroid for prop in predicted_props],
        dtype=np.float32,
    ).reshape(-1, 3)

    # The nearest-neighbor spacing is undefined for one true nucleus.
    if num_ground_truth == 1:
        max_distance = float(single_nucleus_max_distance)

    else:
        distances_ground_truth = cdist(
            ground_truth_centroids,
            ground_truth_centroids,
        )

        # Ignore each object's distance to itself.
        np.fill_diagonal(distances_ground_truth, np.inf)

        nearest_neighbor_distances = np.min(
            distances_ground_truth,
            axis=1,
        )

        max_distance = float(
            np.mean(nearest_neighbor_distances)
        )

    distances = cdist(
        ground_truth_centroids,
        predicted_centroids,
    )

    matches = distances <= max_distance

    # A ground-truth nucleus is considered detected when any predicted
    # centroid lies within the threshold.
    num_detected = np.any(matches, axis=1).sum()

    predicted_count_ratio = (
        num_predicted / num_ground_truth
    )

    detected_fraction = (
        num_detected / num_ground_truth
    )

    return predicted_count_ratio, detected_fraction

