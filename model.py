import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(100, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y = F.relu(self.fc2(x))
        z = self.fc3(y)
        return (x,y,z,100)



net = Net()
print(net)

sm = torch.jit.script(net)
print(sm)
torch.jit.save(sm, 'test.jit')
