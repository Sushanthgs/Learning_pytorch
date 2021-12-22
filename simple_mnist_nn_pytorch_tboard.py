#%%
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter('runs/simple_mnist')

input_size=784
num_classes=10
hidden_size=128
num_epochs=10
batch_size=100
learning_rate=0.001
training_datasets=torchvision.datasets.MNIST(root='./data',train=True,
transform=transforms.ToTensor(),download=True)
testing_datasets=torchvision.datasets.MNIST(root='./data',train=False,
transform=transforms.ToTensor(),download=False)
train_dataloader=torch.utils.data.DataLoader(dataset=training_datasets,
batch_size=batch_size,shuffle=True)
test_dataloader=torch.utils.data.DataLoader(dataset=testing_datasets,
batch_size=batch_size,shuffle=False)
examples=iter(train_dataloader)
samples,labels=examples.next()
print(samples.shape)
img_grid=torchvision.utils.make_grid(samples)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],'gray')
writer.add_image('mnist_images',img_grid)


running_loss=0
running_correct_predictions=0
class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size, hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        return(out)
model=NeuralNet(input_size,hidden_size,num_classes)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
writer.add_graph(model,samples.reshape(-1,28*28))
n_total_steps=len(train_dataloader)
for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_dataloader):
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _, predicted=torch.max(outputs.data,1)
        running_correct_predictions+=(predicted==labels).sum().item()
        if((i+1)%100==0):
            print(f'Epochs:{epoch}, Loss:{loss.item()}')
            writer.add_scalar('training_loss',running_loss/100,epoch*n_total_steps+i)
            writer.add_scalar('accuracy',running_correct_predictions/100,epoch*n_total_steps+i)
            running_correct_predictions=0.0
            running_loss=0.0
writer.close()
sys.exit()
#%%
op_pred_labels=[]
gt_labels=[]
for i,(images,labels) in enumerate(test_dataloader):
    with torch.no_grad():
        t_imgs=images.reshape(-1,28*28).to(device)
        op_preds=model(t_imgs)
        op_pred_labels.append(np.argmax(op_preds.detach(),axis=1))
        gt_labels.append(labels.detach().numpy())
#%%
op_pred_labels_all=np.concatenate([op_pred_labels[i].numpy() for i in range(len(op_pred_labels))],axis=0)
gt_labels_all=np.concatenate([gt_labels[i] for i in range(len(gt_labels))],axis=0)
#%%
plt.imshow(confusion_matrix(gt_labels_all,op_pred_labels_all))
plt.colorbar()