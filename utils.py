import torch
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
from teacher import ResUNet
from dataset import *
import torch.nn as nn

class LRFinder:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.train()

    def range_test(self, train_loader, end_lr=10, num_iter=100, smooth_f=0.05):
        lrs = []
        losses = []
        best_loss = float('inf')
        lr_scheduler = ExponentialLR(self.optimizer, end_lr, num_iter)

        for iteration, (inputs, labels) in enumerate(train_loader):
            if iteration == num_iter:
                break

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Smooth the loss
            if iteration > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]

            if loss < best_loss:
                best_loss = loss

            losses.append(loss.item())
            lrs.append(lr_scheduler.get_last_lr()[0])

            loss.backward()
            self.optimizer.step()
            lr_scheduler.step()

        return lrs, losses

class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, end_lr, num_iter, last_epoch=-1):
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_iter = self.last_epoch + 1
        r = curr_iter / self.num_iter
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]

train_imagespath_dubai= r'DubaiSat2/*/images/*.jpg'
train_maskspath_dubai = r'DubaiSat2/*/masks/*.png'
train_dataset_dubai = CustomDataset(train_imagespath_dubai, train_maskspath_dubai,data = 'dubai', patch_size=128,test_dubai=False)
train_loader_dubai = DataLoader(train_dataset_dubai, batch_size=8, shuffle=True)

model = ResUNet #  model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
criterion = nn.BCEWithLogitsLoss() # your loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr_finder = LRFinder(model, optimizer, criterion, device)
lrs, losses = lr_finder.range_test(train_loader_dubai, end_lr=10, num_iter=100)

plt.plot(lrs, losses)
plt.xscale("log")
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.show()
