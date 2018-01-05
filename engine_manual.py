import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable


ACTIVATIONS = {
    nn.ReLU: trt.infer.ActivationType.RELU,
    nn.Sigmoid: trt.infer.ActivationType.SIGMOID,
    nn.Tanh: trt.infer.ActivationType.TANH
}


POOLING = {
    nn.MaxPool2d: trt.infer.PoolingType.MAX,
    nn.AvgPool2d: trt.infer.PoolingType.AVERAGE
}


class Net(nn.Module):

    # assuming that inputs are 1x3x64x64
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.final = nn.Linear(128 * 16 * 16, 2)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.final(x)
        x = self.sigm(x)
        return x


G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
builder = trt.infer.create_infer_builder(G_LOGGER)

model = Net()

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
builder = trt.infer.create_infer_builder(G_LOGGER)

network = builder.create_network()


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def add_convolution(network, prev_layer, layer):
    num_output = layer.out_channels
    state = layer.state_dict()
    W = state['weight'].numpy().flatten()
    b = state['bias'].numpy().flatten() if 'bias' in state else np.zeros(num_output)
    conv = network.add_convolution(prev_layer, num_output, layer.kernel_size, W, b)
    assert conv
    conv.set_stride(layer.stride)
    conv.set_dilation(trt.infer.DimsHW(layer.dilation))  # NVIDIA, you're drunk, go home
    conv.set_padding(layer.padding)
    return conv.get_output(0)


def add_activation(network, prev_layer, layer):
    if type(layer) not in ACTIVATIONS:
        raise ValueError('Unknown activation type, use one of {}'.format(list(ACTIVATIONS.keys())))
    else:
        act = network.add_activation(prev_layer, ACTIVATIONS[type(layer)])
        assert act
        return act.get_output(0)


def add_pooling(network, prev_layer, layer):
    if type(layer) not in POOLING:
        raise ValueError('Unknown pooling type, use one of {}'.format(list(POOLING.keys())))
    else:
        k = as_tuple(layer.kernel_size)
        s = as_tuple(layer.stride)
        pool = network.add_pooling(prev_layer, POOLING[type(layer)], trt.infer.DimsHW(k))
        assert pool
        pool.set_stride(trt.infer.DimsHW(s))
        return pool.get_output(0)


def add_linear(network, prev_layer, layer):
    num_output = layer.out_features
    state = layer.state_dict()
    W = state['weight'].numpy().flatten()
    b = state['bias'].numpy().flatten() if 'bias' in state else np.zeros(num_output)
    lin = network.add_fully_connected(prev_layer, num_output, W, b)
    assert lin
    return lin.get_output(0)


net = network.add_input("data", trt.infer.DataType.FLOAT, (3, 64, 64))
net = add_convolution(network, net, model.conv1)
net = add_activation(network, net, model.relu1)
net = add_convolution(network, net, model.conv2)
net = add_activation(network, net, model.relu2)
net = add_pooling(network, net, model.pool)
net = add_linear(network, net, model.final)
net = add_activation(network, net, model.sigm)

network.mark_output(net)

builder.set_max_batch_size(1)
builder.set_max_workspace_size(1 << 20)

engine = builder.build_cuda_engine(network)

network.destroy()
builder.destroy()
