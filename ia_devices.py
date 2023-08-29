import torch


class TorchDevices:
    def __init__(self):
        self.cpu = torch.device("cpu")
        self.device = torch.device("cuda") if torch.cuda.is_available() else self.cpu


devices = TorchDevices()
