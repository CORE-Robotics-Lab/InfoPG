from torchvision.models import alexnet
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, CenterCrop


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

    def show_tensor(self, tensor):
        if len(tensor.shape) != 3:
            raise Exception('Tensor shape should be: bxhxw')
        fig, ax = plt.subplots(tensor.shape[0],1, figsize=(20, 10))
        for i in range(0, tensor.shape[0]):
            ax.imshow(tensor[i], cmap='gray')
        plt.show()

    def format_numpy_to_torch(self, nd_array: np.ndarray) -> torch.Tensor:
        #test has shape: batch(10)xheight(457)xwidth(120)xchannel(3)
        test = torch.tensor(nd_array, requires_grad=False, dtype=torch.float64)
        test = test/255.0
        #move tensor of b(0)xh(1)xw(2)xc(3) =-> b(0)xc(3)xh(1)xw(2)
        test = test.permute(0, 3, 1, 2)
        preprocess = Compose([
            Resize(size=75),
        ])

        input_tensor=preprocess(test).float()
        input_tensor = input_tensor[:,0,:,:,]
        #self.show_tensor(input_tensor)
        gpu_tensor = input_tensor.to(self.device)
        return gpu_tensor

    def forward(self, observation: np.ndarray) -> torch.Tensor:
        tensor = self.format_numpy_to_torch(observation)
        num_batches = tensor.shape[0]
        return torch.reshape(tensor, shape=(num_batches, -1)) # output shape is num_batchesx6525

if __name__ == '__main__':
    test_arr = np.random.random((1, 280, 240, 3)) #wxhxc

    encoder = Encoder('cpu')
    print(encoder(test_arr).shape)
    #print(encoder.observation_encoder.eval())
