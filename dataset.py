import os
import cv2
import numpy as np
from patchify import patchify
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import matplotlib.pyplot as plt
import numbers
from numpy.lib.stride_tricks import as_strided


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, limit=None, data = 'mit',test_dubai=True, patch_size=128, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            patch_size (int): Size of the patches to be created.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_patches = []
        self.mask_patches = []
        self.transform = transform
        self.patch_size = patch_size
        if (data == 'mit'):
          self.process_and_append_mit(image_dir, mask_dir, limit)
        else :
          self.process_and_append_dubai(image_dir, mask_dir, limit, test_dubai)
    

    def process_and_append_mit(self, image_dir, mask_dir, limit = None):
            
        def extract_patches(arr, patch_shape=(1000,1000,3), extraction_step=1000):
            arr_ndim = arr.ndim

            if isinstance(patch_shape, numbers.Number):
                patch_shape = tuple([patch_shape] * arr_ndim)
            if isinstance(extraction_step, numbers.Number):
                extraction_step = tuple([extraction_step] * arr_ndim)

            patch_strides = arr.strides

            slices = tuple(slice(None, None, st) for st in extraction_step)
            indexing_strides = arr[slices].strides

            patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                                np.array(extraction_step)) + 1

            shape = tuple(list(patch_indices_shape) + list(patch_shape))
            strides = tuple(list(indexing_strides) + list(patch_strides))

            patches = as_strided(arr, shape=shape, strides=strides)
            return patches
        
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        if limit != None:
            image_files = image_files[0:limit]
        for img_file in image_files:
            large_image = cv2.imread(os.path.join(image_dir, img_file), cv2.IMREAD_COLOR)
            large_image = cv2.cvtColor(large_image, cv2.COLOR_BGR2RGB)

            mask_file = img_file.replace('_sat.jpg', '_mask.png')
            large_mask = cv2.imread(os.path.join(mask_dir, mask_file))
            #large_mask_red_channel = large_mask[:, :, 2]  # Extracting the red channel
            #print(large_image.shape)
            #print(large_mask.shape)
           
            patches_img = extract_patches(large_image, (self.patch_size, self.patch_size, 3), extraction_step=self.patch_size)
            patches_mask = extract_patches(large_mask, (self.patch_size, self.patch_size,1), extraction_step=self.patch_size)
            #exit()
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i, j, 0, :, :, :]
                    single_patch_mask = patches_mask[i, j, 0, :, :,:]
                    #exit()

                    # Convert to binary mask
                    single_patch_mask = (single_patch_mask > 127).astype('uint8')  # Assuming the mask is in 0-255 range

                    # Keep only patches with buildings
                    if np.max(single_patch_mask) != 0:
                        self.image_patches.append(single_patch_img)
                        self.mask_patches.append(single_patch_mask)


    def process_and_append_dubai(self, image_dir= r"G:/KD/DubaiSat2/*/images/*.jpg", mask_dir=r"G:/KD/DubaiSat2/*/masks/*.png", limit=None,test_dubai=True):

          class_labels_R = ["Building", "Road", "Land", "UnLabeled", "Water", "Vegetation"]

          """# Load Dataset"""
          patch_size = self.patch_size #newly added because this function is not seeing the value from init args
          
          list_ims = glob.glob(image_dir) #Linux
          #print(image_dir)
          img_idx =1 # leave an image out from each tile, these images will be used for testing
          image_patches = []
          #test_patches = []
          for i in sorted(list_ims):
              im = cv2.imread(i, 1)
              im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
              im_patches = patchify(im, (patch_size,patch_size, 3), step=patch_size )

              if (test_dubai==True):
                  if 'image_part_00' + str(img_idx) in i:
                      for j in range(im_patches.shape[0]):
                          for k in range(im_patches.shape[1]):
                              image_patches.append(np.squeeze(im_patches[j,k,:,:,:,:], axis = (0,)))
              else:
                  if ('image_part_00' + str(img_idx) in i) ==False:
                      for j in range(im_patches.shape[0]):
                          for k in range(im_patches.shape[1]):
                              image_patches.append(np.squeeze(im_patches[j,k,:,:,:,:], axis = (0,)))

          list_msks = glob.glob(mask_dir) # Linux
          mask_patches = []
          for i in sorted(list_msks):
              im = cv2.imread(i, 1)
              im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
              m_patches = patchify(im, (patch_size,patch_size, 3), step=patch_size )
              #print('HERE!!!!')
              if (test_dubai==True):
                  if 'image_part_00' + str(img_idx) in i:
                    #print('HERE')
                    for j in range(m_patches.shape[0]):
                        for k in range(m_patches.shape[1]):
                            mask_patches.append(np.squeeze(m_patches[j,k,:,:,:,:], axis = (0,)))
              else:
                  if ('image_part_00' + str(img_idx) in i) ==False:
                    #print('HERE')
                    for j in range(m_patches.shape[0]):
                        for k in range(m_patches.shape[1]):
                            mask_patches.append(np.squeeze(m_patches[j,k,:,:,:,:], axis = (0,)))


          # Convert Mask Images into a proper format (Binary Format)

          # Load Red Band only
          tmp_masks = []
          for i in range(len(mask_patches)):
              tmp_masks.append(mask_patches[i][:,:,0])

          masks_unique_vals = [];
          for i in range(len(tmp_masks)):
              vals = list(np.unique(tmp_masks[i]))
              masks_unique_vals.extend(vals)
          masks_unique_vals = np.unique(masks_unique_vals)

          def convert_dubai_masks(im, uv):
              new_mask = np.zeros(im.shape)
              #print(uv.shape)
              #exit()
              new_mask = (im<=uv[1])*1 + (im==uv[2])*0 + (im==uv[3])*0 + (im==uv[4])*0 + (im==uv[5])*0 + (im==uv[6])*0

              
              return new_mask

          Dubai_mask = []
          Dubai_image = []

          for i in range(len(tmp_masks)):
              #print(i)
              current_mask = convert_dubai_masks(tmp_masks[i], masks_unique_vals)
              Dubai_mask_with_buildings = np.max(current_mask) != 0  # Check if the mask has buildings

              if Dubai_mask_with_buildings:
                  #Dubai_mask.append(current_mask)
                  current_mask = (current_mask > 0).astype(np.uint8)
                  self.mask_patches.append(current_mask)
                  #Dubai_image.append(image_patches[i])
                  self.image_patches.append(image_patches[i])
                  #convert list to array
                  #Dubai_mask =np.array(Dubai_mask )
                  #Dubai_mask = (Dubai_mask > 0).astype(np.uint8)
                  #Dubai_image=np.array(Dubai_image)
          if(limit != None): #limit is related to the patches!
              self.mask_patches[0:limit]
              self.image_patches[0:limit]

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, idx):
        image = self.image_patches[idx]
        mask = self.mask_patches[idx]

        if self.transform:
            image = self.transform(image)

        # Convert numpy arrays to PyTorch tensors
        image = torch.from_numpy(image).float().permute(2, 0, 1)  # HWC to CHW
        mask = torch.from_numpy(mask).long().squeeze()  # Remove channel dimension
        #print(image.shape)
        #print(mask.shape)
        return image, mask

def show_images_masks_grid(images, masks, num_images=8):
    """
    Visualizes a grid of images and their corresponding masks.

    Parameters:
    images (tensor): A batch of images in the format [B, C, H, W].
    masks (tensor): A batch of masks corresponding to the images.
    num_images (int): Number of images to display in the grid.
    """
    # Convert tensors to numpy arrays
    images = images.numpy()
    masks = masks.numpy()

    # Create a grid of subplots
    fig, axs = plt.subplots(2, num_images, figsize=(20, 5))

    for i in range(num_images):
        # Convert from CHW to HWC format for displaying
        img = np.transpose(images[i], (1, 2, 0))
        mask = masks[i]  #  single channel
        #print(img.dtype)
        #exit()
        # Display image
        axs[0, i].imshow(img/255)
        axs[0, i].axis('off')

        # Display mask
        axs[1, i].imshow(mask, cmap='gray')
        axs[1, i].axis('off')

    plt.show()

if __name__ == "__main__":

      import torch
      import numpy as np
      from torchvision.utils import make_grid
      from torch.utils.data import DataLoader
      #train_imagespath_mit= r'G:\KD\Massachusetts_dataset\png\train'
      #train_maskspath_mit = r'G:\KD\Massachusetts_dataset\png\train_labels'
      #train_dataset_mit = CustomDataset(train_imagespath_mit, train_maskspath_mit ,data = 'mit', patch_size=128)
      #train_loader_mit = DataLoader(train_dataset_mit, batch_size=8, shuffle=True)

      #test_imagespath_mit= r'G:\KD\Massachusetts_dataset\png\test'
      #test_maskspath_mit= r'G:\KD\Massachusetts_dataset\png\test_labels'
      #test_dataset_mit = CustomDataset(test_imagespath_mit, test_maskspath_mit ,data = 'mit', patch_size=128)
      #test_loader_mit = DataLoader(test_dataset_mit, batch_size=8, shuffle=True)

      #train_imagespath_dubai= r'DubaiSat2/*/images/*.jpg'
      #train_maskspath_dubai = r'DubaiSat2/*/masks/*.png'
      #train_dataset_dubai = CustomDataset(train_imagespath_dubai, train_maskspath_dubai,data = 'dubai', patch_size=128,test_dubai=False)
      #"train_imagespath_dubai" was supplied twice as argument for image and mask directory
      #it has been removed because the function's default argument is enough
      #print(train_dataset_dubai)
      #train_loader_dubai = DataLoader(train_dataset_dubai, batch_size=8, shuffle=True)
      
      test_imagespath_dubai= r'DubaiSat2/*/images/*.jpg'
      test_maskspath_dubai=r'DubaiSat2/*/masks/*.png'
      test_dataset_dubai = CustomDataset(test_imagespath_dubai, test_imagespath_dubai , data = 'dubai', patch_size=128,test_dubai=True) #this was train_dubai but it was giving an error "init() got an unexpected argument"
      test_loader_dubai = DataLoader(test_dataset_dubai, batch_size=8, shuffle=True)

      #train_dataset = CustomDataset(r'G:\KD\Massachusetts_dataset\png\train', r'G:\KD\Massachusetts_dataset\png\train_labels', transform=transforms.ToTensor(), limit = 2)
      #train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
      #val_dataset = CustomDataset(r'G:\KD\Massachusetts_dataset\png\test', r'G:\KD\Massachusetts_dataset\png\test_labels', transform=transforms.ToTensor(), limit = 2)
      #val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


      # Fetch one batch of images and masks
      for images, masks in test_loader_dubai:

          show_images_masks_grid(images, masks, num_images=8)
          break  # Only show the first batch
