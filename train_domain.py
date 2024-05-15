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

# Number of epochs
global max_num_epochs

max_num_epochs = 300


def exponential_function(epoch):
    return 2/(1+np.exp(-10*epoch/max_num_epochs)) - 1
    #return 1.0 - (1.0 - 0.0) * (2.0 ** (-epoch / num_epochs))


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


def save_checkpoint(model, epoch, path="model_checkpoint.pth"):
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, path)



def save_best_checkpoint(model, epoch, avg_iou, path="model_best_checkpoint.pth"):
    global best_iou
    os.makedirs('plots', exist_ok=True)
    if avg_iou > best_iou:
        best_iou = avg_iou
        path = r'plots\\' + path
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},  path)


global best_iou
best_iou = 0

def valid_model(val_loader,model,val_loss_list,val_iou_list,avg_accuracy_list,avg_precision_list,avg_f1_list,epoch,lr_scheduler, criterion):
        val_loss= 0.0
        total_iou, total_accuracy, total_precision, total_f1 = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():

            for images, masks in val_loader:

                images, masks = images.to(device), masks.to(device)

                masks = (masks.squeeze()).int()
                outputs = model(images)
                outputs = outputs.squeeze()
                loss = criterion(outputs, masks.float())

                preds = (torch.sigmoid(outputs) > 0.5).int()
                val_loss += loss.item()
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

        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)

        # Average the metrics
        avg_iou = total_iou /len(val_loader)
        avg_accuracy = total_accuracy /len(val_loader)
        avg_precision = total_precision /len(val_loader)
        avg_f1 = total_f1 /len(val_loader)
        

        val_iou_list.append(avg_iou), avg_accuracy_list.append(avg_accuracy), avg_precision_list.append(avg_precision), avg_f1_list.append(avg_f1)    # Save the best model

        return images, masks, preds, avg_iou


def train_model(train_image_path_source, train_mask_path_source, val_image_path_source, val_mask_path_source, train_image_path_target,train_mask_path_target, val_image_path_target, val_mask_path_target,
                num_epochs, batch_size, learning_rate, patch_size, model, data_='mit', limit = None):


    # Prepare the datasets and dataloaders


    source_dataset = CustomDataset(train_image_path_source, train_mask_path_source, patch_size=patch_size, data=data_, limit=limit)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    source_dataset_val = CustomDataset(val_image_path_source, val_mask_path_source, patch_size=patch_size, data=data_, limit=limit)
    source_val_loader = DataLoader(source_dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)

    target_dataset_val = CustomDataset(val_image_path_target, val_mask_path_target, patch_size=patch_size, data=data_, limit=limit)
    target_loader = DataLoader(target_dataset_val, batch_size=batch_size, shuffle=False)

    target_dataset_train = CustomDataset(train_image_path_target, train_mask_path_target, patch_size=patch_size, data=data_, limit=limit)
    target_loader_train = DataLoader(target_dataset_train, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Teacher optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    # Domain Classifier optimizer and loss function
    #optimizer2 = torch.optim.Adam(domain_classifier.parameters(), lr=learning_rate, weight_decay=0.0001)
    #optimizer2 = torch.optim.Adam(list(model.parameters()) + list(domain_classifier.parameters()), lr=learning_rate2, weight_decay=0.0001)


    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop

    train_loss_list_source = []
    val_loss_list_source = []
    val_iou_list_source = []

    train_loss_list_domain= []
    val_loss_list_target = []
    val_iou_list_target = []

    avg_accuracy_list_source, avg_precision_list_source, avg_f1_list_source = [], [], []
    avg_accuracy_list_target, avg_precision_list_target, avg_f1_list_target = [], [], []
    #num_epochs = 20
    domain_loss_fn = nn.CrossEntropyLoss()
    os.makedirs('plots', exist_ok=True) #newly added because the plots folder gets created in "save_best_checkpoint" whcih happens later
    df = open('plots\\train_log.txt','w')
    for epoch in range(num_epochs):

        model.train()
        print(epoch)
        train_loss_source = 0.0
        train_loss_domain = 0.0

        # Training on source
        #for images, masks in train_loader:
        for (images_s, masks_s), (images_t, _) in zip(source_loader, target_loader_train):
            images_s, masks_s, images_t = images_s.to(device), masks_s.to(device), images_t.to(device)
            #images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images_s)
            outputs = outputs.squeeze()
            masks_s = masks_s.squeeze()

            loss_label = criterion(outputs, masks_s.float())
            loss_label.backward()
            optimizer.step()
            train_loss_source += loss_label.item()

		    # Train domain classifier on both domains
            #combined_data = torch.cat([images_s, images_t], 0)
            alpha = exponential_function(epoch)
            domain_labels_source = torch.zeros(images_s.size(0)).long().cuda()
            domain_labels_target = torch.ones(images_t.size(0)).long().cuda()
            output_source, domain_output_source = model(images_s,alpha ,domain = True)
            output_target, domain_output_target = model(images_t,alpha ,domain = True)
            #print(feature_combined[1].shape)
            #reversed_features = gradient_reversal_layer(feature_combined[1], alpha) #we don't need all of it, just the bottle neck because we want to domain adapt on features only, not the prediction
            
            domain_loss_source = domain_loss_fn(domain_output_source, domain_labels_source)
            domain_loss_target = domain_loss_fn(domain_output_target, domain_labels_target)
            domain_loss = domain_loss_source + domain_loss_target + loss_label.data
            domain_loss.backward()  # Update feature extractor & domain classifier
            optimizer.step()
            

            train_loss_domain += domain_loss.item()


        train_loss_source /= len(source_loader)
        train_loss_list_source.append(train_loss_source)

        train_loss_domain /= max(len(source_loader),len(target_loader))
        train_loss_list_domain.append(train_loss_domain)

        images_source, masks_source, preds_source, avg_iou = valid_model(source_val_loader,model,val_loss_list_source,val_iou_list_source,avg_accuracy_list_source, avg_precision_list_source, avg_f1_list_source,epoch,lr_scheduler, criterion)
        save_best_checkpoint(model, epoch, avg_iou)
        lr_scheduler.step(avg_iou)
        images_target, masks_target, preds_target, _ = valid_model(target_loader,model,val_loss_list_target,val_iou_list_target,avg_accuracy_list_target, avg_precision_list_target, avg_f1_list_target,epoch,lr_scheduler, criterion)

        if (epoch + 1) % 5 == 0:
            # Plot and save figures
            save_plots(epoch, train_loss_list_source, val_loss_list_source,val_iou_list_source,avg_accuracy_list_source, avg_precision_list_source, avg_f1_list_source,folder="plots/source")
            save_plots(epoch, train_loss_list_domain,val_loss_list_target,val_iou_list_target,avg_accuracy_list_target, avg_precision_list_target, avg_f1_list_target,folder="plots/target")

            show_images_masks_predictions_grid(images_source.cpu(), masks_source.cpu(), preds_source.detach().cpu(), epoch, num_images=8,subfolder= 'source')
            show_images_masks_predictions_grid(images_target.cpu(), masks_target.cpu(), preds_target.detach().cpu(), epoch, num_images=8,subfolder= 'target')
        print('*'*50)
        print(f'Epoch  Source[{epoch+1}/{num_epochs}], Alpha {alpha :.4f}, Train Loss: {train_loss_list_source[-1]:.4f}, Val Source Loss: {val_loss_list_source[-1]:.4f}, Val IoU: {val_iou_list_source[-1]:.4f}, Accuracy: {avg_accuracy_list_source[-1]:.4f}, Precision: {avg_precision_list_source[-1]:.4f}, F1 Score: {avg_f1_list_source[-1]:.4f}')
        print(f'Epoch  target[{epoch+1}/{num_epochs}], Alpha {alpha :.4f},Train Loss: {train_loss_list_domain[-1]:.4f}, Val Source Loss: {val_loss_list_target[-1]:.4f}, Val IoU: {val_iou_list_target[-1]:.4f}, Accuracy: {avg_accuracy_list_target[-1]:.4f}, Precision: {avg_precision_list_target[-1]:.4f}, F1 Score: {avg_f1_list_target[-1]:.4f}')

        df.write(f'Epoch  Source[{epoch+1}/{num_epochs}], Train Loss: {train_loss_list_source[-1]:.4f}, Val Source Loss: {val_loss_list_source[-1]:.4f}, Val IoU: {val_iou_list_source[-1]:.4f}, Accuracy: {avg_accuracy_list_source[-1]:.4f}, Precision: {avg_precision_list_source[-1]:.4f}, F1 Score: {avg_f1_list_source[-1]:.4f}\n')
        df.write(f'Epoch  target[{epoch+1}/{num_epochs}], Train Loss: {train_loss_list_domain[-1]:.4f}, Val Source Loss: {val_loss_list_target[-1]:.4f}, Val IoU: {val_iou_list_target[-1]:.4f}, Accuracy: {avg_accuracy_list_target[-1]:.4f}, Precision: {avg_precision_list_target[-1]:.4f}, F1 Score: {avg_f1_list_target[-1]:.4f}\n')
    df.close()


if __name__ == "__main__":

    # Set paths and hyperparameters
    #train_image_path_source = r'./Massachusetts_dataset/png/train'
    #train_mask_path_source = r'./Massachusetts_dataset/png/train_labels'

    #val_image_path_source = r'./Massachusetts_dataset/png/test'
    #val_mask_path_source = r'./Massachusetts_dataset/png/test_labels'

    train_image_path_source = r'./Inria/temp'
    train_mask_path_source = r'./Inria/temp_label'

    val_image_path_source = r'./Inria/test'
    val_mask_path_source = r'./Inria/test_labels'

    ################################################################


    train_image_path_target = r'./DubaiSat2_buildings/train'
    train_mask_path_target = r'./DubaiSat2_buildings/train_labels'
    
    val_image_path_target = r'./DubaiSat2_buildings/test'
    val_mask_path_target = r'./DubaiSat2_buildings/test_labels'



    #val_image_path_target = r'./DubaiSat2_buildings/test'
    #val_mask_path_target = r'./DubaiSat2_buildings/test_labels'

    #train_image_path_target = r'./DubaiSat2_buildings/train'
    #train_mask_path_target = r'./DubaiSat2_buildings/train_labels'

    num_epochs = 200 ##### experiment
    batch_size = 32 ##### it was 4
    learning_rate = 0.00005 #this is for teacher ##### experiments with lr scheduler experiment with less patience
    learning_rate2 = 0.00001  #this is for domain classifier
    #make the scheduler monitor iou instead of loss, and instead of min it will be max
    patch_size = 128
    # already we have it for 256 and 128, 512 
    data = 'mit' #always mit since dubai processing is now the same
    limit = None #start with 50 when doing Dubai

    os.makedirs('plots', exist_ok=True)
    log = open('plots\\parameters.txt','w')
    log.write("domain source = " + train_image_path_source + " \n")
    log.write("domain target = " + val_image_path_target + " \n")
    log.write("num_epochs = " + str(num_epochs) + " \n")
    log.write("batch_size = " + str(batch_size) + " \n")
    log.write("teacher learning_rate = " + str(learning_rate) + " \n")
    log.write("domain classifier learning_rate = " + str(learning_rate2) + " \n")
    log.write("patience = " + str(10) + " \n")
    log.write("data = " + data + " \n")

    log.close()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ResUNet(n_class=1)
    model.to(device)

    # Start training
    train_model(train_image_path_source, train_mask_path_source, val_image_path_source, val_mask_path_source, train_image_path_target, train_mask_path_target, val_image_path_target, val_mask_path_target,
                num_epochs, batch_size, learning_rate, patch_size, model, data_ = data, limit = limit )


