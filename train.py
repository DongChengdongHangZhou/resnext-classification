import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
from network import resnext50_32x4d
from functions import BatchTimer, accuracy, pass_epoch

torch.cuda.empty_cache()
data_dir_train = './data/train'
data_dir_val = './data/val'
batch_size = 8
epochs = 25
workers = 0
save_every = 30
save_checkpoints_path = './checkpoints/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

resnet = resnext50_32x4d(pretrained=None, num_classes = 2).to(device)

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])


trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
])
dataset_train = datasets.ImageFolder(data_dir_train, transform=trans)
dataset_val = datasets.ImageFolder(data_dir_val, transform=trans)
testtest = dataset_train[0]
img_inds_train = np.arange(len(dataset_train))
img_inds_val = np.arange(len(dataset_val))
np.random.shuffle(img_inds_train)
np.random.shuffle(img_inds_val)

train_loader = DataLoader(
    dataset_train,
    num_workers=workers,
    batch_size=batch_size,
    shuffle=True
)
val_loader = DataLoader(
    dataset_val,
    num_workers=workers,
    batch_size=batch_size,
    shuffle=True
)

loss_fn = torch.nn.CrossEntropyLoss()

metrics = {
    'fps': BatchTimer(),
    'acc': accuracy
}


print('\n\nInitial')
print('-' * 10)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    pass_epoch(
        epoch=epoch, model=resnet, loss_fn=loss_fn, loader=train_loader, optimizer=optimizer, scheduler=scheduler,
        save_every=save_every,batch_metrics=metrics, show_running=True, device=device
    )

    if epoch%4 == 0:
        torch.save({
            'inceptionNet': resnet.state_dict()
        },save_checkpoints_path+'inceptionNet'+'%04d.pth'%epoch
        )

    resnet.eval()
    pass_epoch(
        epoch=epoch, model=resnet, loss_fn=loss_fn, loader=val_loader, optimizer=None,scheduler=None,save_every=save_every,
        batch_metrics=metrics, show_running=True,device=device
    )