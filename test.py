import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DiabetesSet(Dataset):
    def __init__(self, path):
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesSet('./dataset/diabetes.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=16,
                          shuffle=True,
                          num_workers=0)

class DiabetesModel(torch.nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = DiabetesModel()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

loss_list = []
for epoch in range(1000):
    current_loss = 0.0
    count  = 0
    for i, data in enumerate(train_loader, 0):
        data, labels = data
        y_pred = model(data)
        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        current_loss += loss.item()
        count = i
    loss_list.append(current_loss/count)

xx = np.arange(1, len(loss_list) + 1, 1)
plt.plot(xx, loss_list)
plt.show()
