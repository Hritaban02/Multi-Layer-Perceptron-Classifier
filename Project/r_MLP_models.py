import torch.nn as nn


# Model1 : 0 hidden layers
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.linear3 = nn.Linear(2, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear3(x)
        return x


# Model2: 1 hidden layer with 2 nodes
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear3 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x


# Model3: 1 hidden layer with 6 nodes
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.linear1 = nn.Linear(2, 6)
        self.linear3 = nn.Linear(6, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x


# Model4: 2 hidden layers with 2 and 3 nodes respectively
class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        # an affine operation: y = Wx + b
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 3)
        self.linear3 = nn.Linear(3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x


# Model5: 5 hidden layers with 3 and 2 nodes respectively
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        # an affine operation: y = Wx + b
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = nn.Linear(3, 2)
        self.linear3 = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x
