import unet
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# np.set_printoptions(threshold=np.inf)

class ImageDataset(Dataset):
    def __init__(self, real_dir, sim_dir, transform=None):
        self.real_dir = real_dir
        self.sim_dir = sim_dir
        self.transform = transform
        self.real_images = sorted(os.listdir(real_dir))
        self.sim_images = sorted(os.listdir(sim_dir))

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        real_path = os.path.join(self.real_dir, self.real_images[idx])
        sim_path = os.path.join(self.sim_dir, self.sim_images[idx])

        real_image = Image.open(real_path).convert("L")
        # print(np.array(real_image.getdata()))
        sim_image = Image.open(sim_path).convert("L")

        if self.transform:
            real_image = self.transform(real_image)
            sim_image = self.transform(sim_image)

        return real_image, sim_image

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

# Images must be of a size power of 2.
real_dir = "images/real"
sim_dir = "images/simulated"

dataset = ImageDataset(real_dir, sim_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = unet.UNet(1)

# Use multiple GPUs if available. Otherwise there is no sufficient memory
if torch.cuda.device_count() > 1:
    # in device_ids you indicate the number of GPUs to use, in this case, 2
    model = nn.DataParallel(model, device_ids=[0, 1])

model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    for real_images, sim_images in dataloader:
        real_images = real_images.to(device)
        sim_images = sim_images.to(device)
        optimizer.zero_grad()
        outputs = model(real_images)
        loss = criterion(outputs, sim_images)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
# print(outputs[0])
output = outputs[0].detach().cpu().numpy().T
plt.imsave("entrenada.png", np.clip(output, 0, 1))
# plt.imshow(output, cmap="binary")
# plt.show()
