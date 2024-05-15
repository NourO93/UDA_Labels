import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import *
from teacher import *

import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(42)


def exponential_function(epoch):
    return 1.0 - (1.0 - 0.0) * (2.0 ** (-epoch / num_epochs))

def show_images_masks_predictions_grid(images, masks, predictions, epoch, num_images=8,subfolder= 'source'):


    """
    Visualizes a grid of images, their corresponding masks, and predictions.

    Parameters:
    images (tensor): A batch of images in the format [B, C, H, W].
    masks (tensor): A batch of ground truth masks corresponding to the images.
    predictions (tensor): A batch of predicted masks corresponding to the images.
    num_images (int): Number of images to display in the grid.
    """


    masks=masks.unsqueeze(1)
    predictions=predictions.unsqueeze(1)
    num_images = min(num_images, images.size(0))


    # Convert tensors to numpy arrays
    images_np = images.numpy()
    masks_np = masks.numpy()
    predictions_np = predictions.numpy()

    # Normalize image values to [0, 1] if not already
    if np.max(images_np) > 1:
        images_np = images_np / 255.0

    # Create a grid of subplots
    fig, axs = plt.subplots(3, num_images, figsize=(20, 7))
    epoch_dir = os.path.join('plots',subfolder ,'vis')
    os.makedirs(epoch_dir, exist_ok=True)

    for i in range(num_images):
        # Convert from CHW to HWC format for displaying
        img = np.transpose(images_np[i], (1, 2, 0))

        mask = masks_np[i][0]  # Assuming single channel
        prediction = predictions_np[i][0]  # Assuming single channel

        # Display image
        axs[0, i].imshow(img)
        axs[0, i].axis('off')

        # Display mask
        axs[1, i].imshow(mask, cmap='gray')
        axs[1, i].axis('off')

        # Display prediction
        axs[2, i].imshow(prediction, cmap='gray')
        axs[2, i].axis('off')

    plt.tight_layout()

    plot_filename = os.path.join(epoch_dir, f'epoch-vis_{epoch}_plot.png')
    plt.savefig(plot_filename)
    plt.close()


def save_plots(epoch, train_loss_list, val_loss_list, val_iou_list, accuracy, precision, f1, folder="plots/source"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.title('Train Loss')

    plt.subplot(2, 3, 2)
    plt.plot(val_loss_list, label='Val Loss')
    plt.title('Validation Loss')

    plt.subplot(2, 3, 3)
    plt.plot(val_iou_list, label='IoU')
    plt.title('IoU')

    plt.subplot(2, 3, 4)
    plt.plot(accuracy, label='Accuracy')
    plt.title('Accuracy')

    plt.subplot(2, 3, 5)
    plt.plot(precision, label='Precision')
    plt.title('Precision')

    plt.subplot(2, 3, 6)
    plt.plot(f1, label='F1 Score')
    plt.title('F1 Score')

    plt.tight_layout()
    plt.savefig(f"{folder}/epoch_{epoch+1}.png")
    plt.close()

# preds and labels are PyTorch tensors

def calculate_iou(preds, labels):
    #print("pred shape: ",preds.shape)
    #print("pred min: ",preds.min())
    #print("pred max: ",preds.max())
    #print("label shape: ",labels.shape)
    #print("label min: ",labels.min())
    #print("label max: ",labels.max())
    intersection = (preds & labels).float().sum((1, 2))
    union = (preds | labels).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def calculate_accuracy(preds, labels):
    correct = (preds == labels).float().sum()
    total = preds.numel()
    return (correct / total).item()

def calculate_precision(preds, labels):
    true_positives = ((preds == 1) & (labels == 1)).float().sum()
    predicted_positives = (preds == 1).float().sum()
    precision = true_positives / (predicted_positives + 1e-6)
    return precision.item()

def calculate_f1_score(preds, labels):
    true_positives = ((preds == 1) & (labels == 1)).float().sum()
    predicted_positives = (preds == 1).float().sum()
    actual_positives = (labels == 1).float().sum()
    precision = true_positives / (predicted_positives + 1e-6)
    recall = true_positives / (actual_positives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1.item()



def test(val_loader,model):
        val_loss= 0.0
        total_iou, total_accuracy, total_precision, total_f1 = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():

            for images, masks in val_loader:
                device = 'cuda'
                images, masks = images.to(device), masks.to(device)

                masks = (masks.squeeze()).int()
                outputs = model(images)
                outputs = outputs.squeeze()

                preds = (torch.sigmoid(outputs) > 0.5).int()
                #val_loss += loss.item()
                if len(preds.shape) < 3:
                    preds=preds.unsqueeze(0)
                    masks=masks.unsqueeze(0)

                try:

                    total_iou += calculate_iou(preds, masks)
                    total_accuracy += calculate_accuracy(preds, masks)
                    total_precision += calculate_precision(preds, masks)
                    total_f1 += calculate_f1_score(preds, masks)
                except:
                    exit()

        #val_loss /= len(val_loader)
        #val_loss_list.append(val_loss)

        # Average the metrics
        avg_iou = total_iou /len(val_loader)
        avg_accuracy = total_accuracy /len(val_loader)
        avg_precision = total_precision /len(val_loader)
        avg_f1 = total_f1 /len(val_loader)
        

       # val_iou_list.append(avg_iou), avg_accuracy_list.append(avg_accuracy), avg_precision_list.append(avg_precision), avg_f1_list.append(avg_f1)    # Save the best model

        return avg_iou, avg_accuracy, avg_precision, avg_f1

#MIT Network --> MIT
path = 'G:\KD\MIT\plots_22022024_scheduler_iou\model_best_checkpoint.pth'# 'G:\KD\Dubai\plots_24012024_Dubai\model_best_checkpoint.pth'
weight = torch.load(path)
model = ResUNet(n_class=1)
model.cuda()
model.load_state_dict(weight['model_state_dict'])
val_dataset = CustomDataset(r'G:\KD\Massachusetts_dataset\png\test', r'G:\KD\Massachusetts_dataset\png\test_labels', patch_size=256, limit = 2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
avg_iou, avg_acc, avg_pr, avg_f = test(val_loader, model)
print('### MIT Network --> MIT ###')
print('avg iou: ', avg_iou)
print('avg acc: ', avg_acc)
print('avg pr: ', avg_pr)
print('avg f: ', avg_f)
print('######################################################')

##################################################################################
#MIT Network --> Dubai

path = 'G:\KD\MIT\plots_22022024_scheduler_iou\model_best_checkpoint.pth'
weight = torch.load(path)
model = ResUNet(n_class=1)
model.cuda()
model.load_state_dict(weight['model_state_dict'])
val_dataset = CustomDataset(r'G:\KD\DubaiSat2_buildings\test', r'G:\KD\DubaiSat2_buildings\test_labels', patch_size=256, limit = 2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
avg_iou, avg_acc, avg_pr, avg_f = test(val_loader, model)
print('### MIT Network --> Dubai ###')
print('avg iou: ', avg_iou)
print('avg acc: ', avg_acc)
print('avg pr: ', avg_pr)
print('avg f: ', avg_f)
print('######################################################')


save_plots(epoch, train_loss_list, val_loss_list, val_iou_list, accuracy, precision, f1, folder="plots/source"):


