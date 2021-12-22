#%%
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

transform_n=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0,0,0), (1,1,1))])
training_datasets=torchvision.datasets.CIFAR10(root='./data',train=True,
transform=transform_n,download=True)
testing_datasets=torchvision.datasets.CIFAR10(root='./data',train=False,
transform=transform_n,download=False)
#%%
num_classes=10
batch_size=32

train_dataloader=torch.utils.data.DataLoader(dataset=training_datasets,
batch_size=batch_size,shuffle=True)
test_dataloader=torch.utils.data.DataLoader(dataset=testing_datasets,
batch_size=batch_size,shuffle=False)
examples=iter(train_dataloader)
samples,labels=examples.next()
#%%
classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.bn1=nn.BatchNorm2d(num_features=3)
        self.con1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5)
        self.p1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn2=nn.BatchNorm2d(num_features=32)
        self.con2=nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.p2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.fc1=nn.Linear(in_features=32*5*5, out_features=120)
        self.fc2=nn.Linear(in_features=120,out_features=60)
        self.fc3=nn.Linear(in_features=60,out_features=10)        
    def forward(self,x):
        x=self.bn1(x)
        x=self.con1(x)
        x=F.relu(x)
        x=self.p1(x)
        x=self.bn2(x)
        x=self.con2(x)
        x=F.relu(x)
        x=self.p2(x)
        x=self.bn3(x)
        x=x.view(-1,32*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return(x)

model=ConvNet()
learning_rate=0.001
criterion=nn.CrossEntropyLoss()
num_epochs=20
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
n_total_steps=len(train_dataloader)
for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_dataloader):
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if((i+1)%500==0):
            print(f'Epochs:{epoch}, Loss:{loss.item()}')
#%%
op_pred_labels=[]
gt_labels=[]
for i,(images,labels) in enumerate(test_dataloader):
    with torch.no_grad():
        t_imgs=images.to(device)
        op_preds=model(t_imgs)
        op_pred_labels.append(np.argmax(op_preds.detach(),axis=1))
        gt_labels.append(labels.detach().numpy())
op_pred_labels_all=np.concatenate([op_pred_labels[i].numpy() for i in range(len(op_pred_labels))],axis=0)
gt_labels_all=np.concatenate([gt_labels[i] for i in range(len(gt_labels))],axis=0)
#%%
plt.imshow(confusion_matrix(gt_labels_all,op_pred_labels_all))
plt.colorbar()

