
from numpy import testing
import torch.cuda
import numpy as np
import time
import array as arr
from datetime import datetime
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    num_workers = 0
    load_data.batch_size = 64
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    load_data.train_loader = torch.utils.data.DataLoader(train_data, 
                                batch_size=load_data.batch_size, num_workers=num_workers, pin_memory=True)
    load_data.test_loader = torch.utils.data.DataLoader(test_data, 
                                batch_size=load_data.batch_size, num_workers=num_workers, pin_memory=True)
    
def visualize():
    dataiter = iter(load_data.train_loader)
    visualize.images, labels = dataiter.next()
    visualize.images = visualize.images.numpy()
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(load_data.batch_size):
        ax = fig.add_subplot(2, load_data.batch_size/2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(visualize.images[idx]), cmap='gray')
        ax.set_title(str(labels[idx].item()))
    #plt.show()

def fig_values():
    img = np.squeeze(visualize.images[1])
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y],2) if img[x][y] !=0 else 0
            ax.annotate(str(val), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y]<thresh else 'black')
    #plt.show()


load_data()
visualize()
fig_values()
class NeuralNet(nn.Module):
    def __init__(self, gpu = True):
        super(NeuralNet, self ).__init__()
        self.fc1 = nn.Linear(28 * 28, 750).to(device)
        self.fc2 = nn.Linear(750, 10).to(device)

    def forward(self, x):
        x = x.view(-1, 28 * 28).to(device)
        x = F.relu(self.fc1(x)).to(device)
        x = self.fc2(x).to(device)
        return x.to(device)

#has antioverfit 
def training():
    training.criterion = nn.CrossEntropyLoss().to(device)
    optimizer= torch.optim.SGD(model.parameters(), 
                lr=0.003, weight_decay= 1e-5, momentum = .9, nesterov = True)

    n_epochs = 1000   
    a = np.float64([9,9,9,9,9]) #antioverfit  
    testing_loss = 0.0
    for epoch in range(n_epochs) :
        if(testing_loss <= a[4]): # part of anti overfit
            train_loss = 0.0        
            testing_loss = 0.0
            model.train().to(device)
            for data, target in load_data.train_loader:
                optimizer.zero_grad()
                data = data.to(device) #gpu
                target = target.to(device) #gpu
                output = model(data).to(device)
                loss = training.criterion(output, target).to(device)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*data.size(0)               
            train_loss = train_loss/len(load_data.train_loader.dataset)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))            
            model.eval().to(device)  # Gets Validation loss 
            train_loss = 0.0        
            with torch.no_grad():
                for data, target in load_data.test_loader:   
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data).to(device)
                    loss = training.criterion(output, target).to(device)
                    testing_loss += loss.item()*data.size(0)
            testing_loss = testing_loss / len(load_data.test_loader.dataset)          
            print('Validation loss = ' , testing_loss)           
            a = np.insert(a,0,testing_loss) # part of anti overfit           
            a = np.delete(a,5)
    print('Validation loss = ' , testing_loss) 
    torch.save(model.state_dict(), "mymodel.pth")  
    torch.save(model, "mymodel01.pth")

def evalution():
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval().to(device)

    for data, target in load_data.test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data).to(device)
        loss = training.criterion(output, target).to(device)
        test_loss += loss.item()*data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred))).to(device)
        for i in range(load_data.batch_size):
            try:
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            except IndexError:
                break

    test_loss = test_loss/len(load_data.test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' )

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


model = NeuralNet().to(device)
summary(model, input_size=(1, 28, 28))
training()
evalution()


