import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1],[2],[3]])
y_data = torch.Tensor([[2],[4],[6]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w is {}, b is {}".format(model.linear.weight.item(), model.linear.bias.item()))
