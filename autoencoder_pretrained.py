import os
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1), 
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1), #8x4x4
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1), #8x8x8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def _default_weights_path() -> str:
    return os.path.join(os.path.dirname(__file__), 'params.pkl')


def create_autoencoder(device='cpu', weights_path=None):
    """
    Create the pretrained AE on the requested device.

    On macOS, the main CNN can run on MPS while the AE stays on CPU to avoid
    cross-backend incompatibilities and memory pressure.
    """
    if weights_path is None:
        weights_path = _default_weights_path()

    device = torch.device(device)
    model = Autoencoder().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model