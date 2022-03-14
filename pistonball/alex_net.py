from torchvision.models import alexnet
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

class SpecificCrop:
    def __init__(self, top, left, h, w):
        self.top = top
        self.left = left
        self.w = w
        self.h = h

    def __call__(self, x):
        #x shld have shape: bxcxhxw or 3x457x120
        return x[:,:, self.top:self.top+self.h, self.left:self.left+self.w]

class Encoder(nn.Module):
    def __init__(self, device):
        super(Encoder, self).__init__()
        self.observation_encoder = alexnet(pretrained=True)
        self.observation_encoder.classifier = self.observation_encoder.classifier[0:4]
        self.observation_encoder.eval()
        self.device = device
        self.observation_encoder.to(device)
        for param in self.observation_encoder.parameters():
            param.requires_grad_(False)

    def show_tensor(self, tensor, cmap=None):
        if len(tensor.shape) != 4:
            raise Exception('Tensor shape should be: bxcxhxw')
        fig, ax = plt.subplots(1, tensor.shape[0], figsize=(20, 10))
        if cmap is None:
            for i in range(0, tensor.shape[0]):
                ax[i].imshow(tensor[i].permute(1, 2, 0))
        else:
            for i in range(0, tensor.shape[0]):
                ax[i].imshow(tensor[i].permute(1,2,0), cmap=cmap)
        plt.show()

    def format_numpy_to_torch(self, nd_array: np.ndarray) -> torch.Tensor:
        #test has shape: batch(10)xheight(457)xwidth(120)xchannel(3)
        test = torch.tensor(nd_array, requires_grad=False, dtype=torch.float64)
        test = test/255.0
        mean=[0.83102926,0.88689262,0.92050735]
        std=[0.27236169,0.18556092,0.15584284]
        #move tensor of b(0)xh(1)xw(2)xc(3) =-> b(0)xc(3)xh(1)xw(2)
        test = test.permute(0, 3, 1, 2)
        preprocess = Compose([
            SpecificCrop(top=457 - 120, left=0, h=120, w=120),
            Resize((224, 224)),
            Normalize(mean=mean, std = std),
        ])
        input_tensor=preprocess(test).float()
        #self.show_tensor(input_tensor, cmap='Greys')
        gpu_tensor = input_tensor.to(self.device)
        return gpu_tensor

    def forward(self, observation: np.ndarray) -> torch.Tensor:
        tensor = self.format_numpy_to_torch(observation)
        x = self.observation_encoder(tensor)
        return x

if __name__ == '__main__':
    test_arr = np.random.random((457, 120, 3)) #wxhxc

    encoder = Encoder('cpu')
    print(encoder.observation_encoder.eval())
