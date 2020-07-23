from torch.utils.data import Dataset
import torch
from pathlib import Path
import numpy as np
import random


class FlyingMNIST(Dataset):

    def __init__(self, datafile, split, limit=None):
        npz = np.load(Path("data") / (datafile + f"_{split}.npz"))
        self.images = npz['images'].astype(np.float32) / 255
        self.targets = npz['targets'].astype(np.float32)
        if limit is not None:
            self.images = random.choice(self.images, limit)
            self.targets = random.choice(self.targets, limit)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        retval = {
            'image': self.images[idx, :, :],
            'target': self.targets[idx, :]
        }
        return retval


def test_FlyingMNIST():
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    dataset = FlyingMNIST("floating_mnist_quadrant", "train")
    batch_size = 12
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    print("Length of dataset:", len(dataset))

    for i, datasample in enumerate(dataloader):
        expand_size = (2, 1, 64, 64)
        coord0 = torch.linspace(0.0, 255, expand_size[2]).view(1, -1, 1).expand(expand_size).numpy().astype(np.uint8)
        coord1 = torch.linspace(0.0, 255, expand_size[3]).view(1, 1, -1).expand(expand_size).numpy().astype(np.uint8)
        for j in range(batch_size):
            image = Image.new('RGB', (64, 64))
            image.paste(Image.fromarray(datasample['image'][j].numpy() * 255))
            target = datasample['target'][j]
            draw = ImageDraw.Draw(image)
            # draw.rectangle([(max(target[1] - 14, 0), max(target[0] - 14, 0)), (min(target[1] + 14, 64), min(target[0] + 14, 64))], outline=(0, 255, 0))
            plt.imshow(image)
            plt.show()
        break


if __name__ == "__main__":
    test_FlyingMNIST()
