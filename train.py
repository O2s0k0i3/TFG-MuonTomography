import unet
import optparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import shutil

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
        sim_image = Image.open(sim_path).convert("L")

        if self.transform:
            real_image = self.transform(real_image)
            sim_image = self.transform(sim_image)

        return real_image, sim_image

def main(img_num):
    # Images must be of a size power of 2.
    real_dir = "images/real"
    sim_dir = "images/simulated"

    train_dataset = ImageDataset(f"{real_dir}/train", f"{sim_dir}/train", transform=transforms.ToTensor())

    val_dataset = ImageDataset(f"{real_dir}/eval", f"{sim_dir}/eval", transform=transforms.ToTensor())

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = unet.UNet(1)

    # Use multiple GPUs if available. Otherwise there is no sufficient memory
    if torch.cuda.device_count() > 1:
        # in device_ids you indicate the number of GPUs to use, in this case, 2
        model = nn.DataParallel(model)

    model.to(device)

    lr = 1e-5
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    output_path = f"output_{img_num}"
    Path(output_path).mkdir(exist_ok=True)
    [os.remove(f"{output_path}/{j}") for j in [i for i in os.listdir(output_path) if i.endswith(".png")]]
    file = open(f"{output_path}/loss.txt", "w")

    epochs = 25
    for epoch in range(epochs):
        for real_images, sim_images in train_loader:
            real_images = real_images.to(device)
            sim_images = sim_images.to(device)
            optimizer.zero_grad()
            outputs = model(real_images)
            loss = criterion(outputs, sim_images)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{epochs}:')
        file.write(f'Epoch {epoch+1}/{epochs}:\n')
        print(f'    Loss: {loss.item():.4f}')
        file.write(f'    Loss: {loss.item():.4f}\n')
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for real_images, sim_images in val_loader:
                real_images = real_images.to(device)
                sim_images = sim_images.to(device)
                outputs = model(real_images)
                loss = criterion(outputs, sim_images)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f'    Validation Loss: {val_loss:.4f}')
        file.write(f'    Validation Loss: {val_loss:.4f}\n')
    file.close()

    input_img = val_dataset[img_num][0].unsqueeze(0).to(device)
    with torch.no_grad():
        image = model(input_img)[0].detach().cpu().numpy().T
    image -= np.min(image)
    output = (image * 255/np.max(image)).astype(np.uint8)

    params_file = open(f"{output_path}/params.txt", "w")
    params_file.write(f"Batch size: {batch_size}\n")
    params_file.write(f"Learning rate: {lr}\n")
    params_file.write(f"Epochs: {epochs}\n")
    params_file.close()
    shutil.copy(f"{real_dir}/eval/real_{img_num}.png", output_path)
    shutil.copy(f"{sim_dir}/eval/sim_{img_num}.png", output_path)
    transforms.ToPILImage(mode="L")(output).save(f"{output_path}/output_{img_num}.png")

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('--img')
    opts, args = parser.parse_args()
    main(int(opts.img))
