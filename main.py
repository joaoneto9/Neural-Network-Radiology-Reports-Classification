from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets. 
training_data = datasets.ImageFolder(
    root="radiology-reports",
    transform=ToTensor()
)

data_loader = DataLoader(training_data, batch_size=32, shuffle=True)

for x,y in data_loader:
    print(f"shape of X [N, C, H, W]: {x.shape}")
    print(f"Shpape of Y: {y.shape} {y.dtype}")
    break