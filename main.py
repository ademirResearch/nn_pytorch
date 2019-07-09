import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return x


criterion = nn.MSELoss()


net = Net()
# net.cuda()
# print(net)

# print(list(net.parameters()))

# input_tensor = Variable(torch.randn(1, 1, 1), requires_grad=True)
input_tensor = torch.randn(1, 1, 1)

out = net(input_tensor)


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.5)

data = [(1, 3), (2, 6), (3, 9), (4, 12), (5, 15), (6, 18), (7, 21)]

for epoch in range(300):
    for i, data2 in enumerate(data):
        X, Y = iter(data2)
        X, Y = torch.FloatTensor([X]), torch.FloatTensor([Y])
        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print("Epoch {} - loss: {}".format(epoch, loss.data))

print(list(net.parameters()))

print("Prediction")
x = torch.tensor([[[7]]], dtype=torch.float32)
print(net(x))

print("Prediction")
y = torch.tensor([[[11]]], dtype=torch.float32)
print(net(y))








