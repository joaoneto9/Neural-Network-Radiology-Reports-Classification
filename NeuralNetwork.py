import torch
import torch.nn as nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# To create a Neural Network 
#   -> create a class that inherits from nn.Modules ("herda")
#   -> define the layers of the netural network ("define as camadas da rede" -> construtor) ]
#   -> define how the data will pass through the network ("como os dados vão fluir" -> foward)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten
        self.linear_rule_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def foward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
    
# se tiver algo mais rapido para treinamento como: CUDA > MPS > MTIA > XPU > CPU
model = NeuralNetwork().to(device)
print(model)
    
