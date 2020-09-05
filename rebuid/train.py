import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

# Data parameters
data_folder = "./"  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [
    80000,
    100000,
]  # decay learning rate after these many iterations
decay_lr_to = (
    0.1  # decay learning rate to this fraction of the existing learning rate
)
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()


    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        images = images.to(device) # batch_size (N, 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        predicted_locs, predicted_scores = model(images) # (N, 8732, 4), (N, 8732, n_classes)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )
    del (
        predicted_locs,
        predicted_scores,
        images,
        boxes,
        labels,
    )  # free some memory since their histories may be stored


def main():
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith(".bias"):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(
            params=[{"params": biases, "lr": 2 * lr}, {"params": not_biases}]
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        ) 

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    train_dataset = PascalVOCDataset(data_folder, split="train", keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True,
    )

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)