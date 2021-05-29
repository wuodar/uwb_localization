from torch.nn import Module, Linear, ReLU

class LinearModel(Module):
    def __init__(self):
        super().__init__()        

        self.fc1 = Linear(2, 32)
        self.fc2 = Linear(32, 16)
        self.fc3 = Linear(16, 8)
        self.fc4 = Linear(8, 2)
        self.relu = ReLU()

    def forward(self, x):
        batch_size, _ = x.size()
        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        
        return x