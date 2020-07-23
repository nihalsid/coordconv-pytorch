import torch
import torch.nn as nn
import torch.nn.functional as F
from util import get_embedder_nerf

class AddCoordinates(object):

    def __init__(self, with_r=False):
        self.with_r = with_r

    def __call__(self, image):
        batch_size, _, image_height, image_width = image.size()

        y_coords = 2.0 * torch.arange(image_height).unsqueeze(
            1).expand(image_height, image_width) / (image_height - 1.0) - 1.0
        x_coords = 2.0 * torch.arange(image_width).unsqueeze(
            0).expand(image_height, image_width) / (image_width - 1.0) - 1.0

        coords = torch.stack((y_coords, x_coords), dim=0)

        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = rs / torch.max(rs)
            rs = torch.unsqueeze(rs, dim=0)
            coords = torch.cat((coords, rs), dim=0)

        coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1)
        image = torch.cat((coords.to(image.device), image), dim=1)
        return image


class SmallNetwork(nn.Module):

    def __init__(self, coords=None):
        super().__init__()
        self.coordinate_adder = AddCoordinates(with_r=True)
        self.embedder, self.embedder_out_dim = get_embedder_nerf(5, input_dims=2, i=0)
        self.coords = coords
        extra_dims = 0
        if coords == 'coord':
            extra_dims = 3
        elif coords == 'pe':
            extra_dims = self.embedder_out_dim
        print('Extra coords in first layer:', extra_dims)
        self.conv = nn.Sequential(
            nn.Conv2d(1 + extra_dims, 8, (1, 1), stride=1, padding=0),
            nn.Conv2d(8, 8, (1, 1), stride=1, padding=0),
            nn.Conv2d(8, 8, (1, 1), stride=1, padding=0),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1),
            nn.Conv2d(8, 2, (3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(64, 64), stride=(64, 64)),
        )

    def forward(self, x):
        if self.coords == 'coord':
            x = self.coordinate_adder(x)
        elif self.coords == 'pe':
            expand_size = x.size()
            coord0 = torch.linspace(-1.0, 1.0, x.size(2)).to(x.device).view(-1, 1).expand(expand_size)
            coord1 = torch.linspace(-1.0, 1.0, x.size(3)).to(x.device).view(1, -1).expand(expand_size)
            coords = torch.stack((coord0, coord1), dim=1).squeeze(2)
            coords = self.embedder(coords)
            x = torch.cat((coords, x), dim=1)
        x = self.conv(x)
        return x.squeeze(3).squeeze(2)


class UniformSplit(nn.Module):

    def __init__(self, coords=None):
        super().__init__()
        self.coordinate_adder = AddCoordinates(with_r=True)
        self.embedder, self.embedder_out_dim = get_embedder_nerf(5, input_dims=2, i=0)
        self.coords = coords
        extra_dims = 0
        if coords == 'coord':
            extra_dims = 3
        elif coords == 'pe':
            extra_dims = self.embedder_out_dim
        print('Extra coords in first layer:', extra_dims)
        self.conv = nn.Sequential(
            nn.Conv2d(1 + extra_dims, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
        )
        self.linear0 = nn.Linear(8 * 8 * 16 , 64)
        self.linear1 = nn.Linear(64, 2)

    def forward(self, x):
        if self.coords == 'coord':
            x = self.coordinate_adder(x)
        elif self.coords == 'pe':
            expand_size = x.size()
            coord0 = torch.linspace(-1.0, 1.0, x.size(2)).to(x.device).view(-1, 1).expand(expand_size)
            coord1 = torch.linspace(-1.0, 1.0, x.size(3)).to(x.device).view(1, -1).expand(expand_size)
            coords = torch.stack((coord0, coord1), dim=1).squeeze(2)
            coords = self.embedder(coords)
            x = torch.cat((coords, x), dim=1)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.linear0(x))
        x = self.linear1(x)
        return x


class QuadrantSplit(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, (5, 5), stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 1), stride=1, padding=0),
            nn.BatchNorm2d(16, momentum=0.9, eps=1e-5),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16, momentum=0.9, eps=1e-5),
            nn.Conv2d(16, 16, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d((2, 2))
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    test_tensor = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    model = SmallNetwork(coords='coord')
    out = model(test_tensor)
    print(out.shape)
