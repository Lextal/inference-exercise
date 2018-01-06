import tensorrt as trt
import numpy as np

from torch import nn


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class ModuleParser:

    ACTIVATIONS = {
        nn.ReLU: trt.infer.ActivationType.RELU,
        nn.Sigmoid: trt.infer.ActivationType.SIGMOID,
        nn.Tanh: trt.infer.ActivationType.TANH
    }

    POOLINGS = {
        nn.MaxPool2d: trt.infer.PoolingType.MAX,
        nn.AvgPool2d: trt.infer.PoolingType.AVERAGE
    }

    def __init__(self):
        self.mapping = {
            nn.Sequential: self.__sequential__,

            nn.Conv2d: self.__convolution__,

            nn.ReLU: self.__activation__,
            nn.Tanh: self.__activation__,
            nn.Sigmoid: self.__activation__,

            nn.AvgPool2d: self.__pooling__,
            nn.MaxPool2d: self.__pooling__
        }

    @staticmethod
    def __input__(network, num_channels=3, height=64, width=64):
        data = network.add_input("data", trt.infer.DataType.FLOAT, (num_channels, height, width))
        assert data
        return data

    @staticmethod
    def __convolution__(network, prev_layer, layer):
        num_output = layer.out_channels
        state = layer.state_dict()
        W = state['weight'].numpy().flatten()
        b = state['bias'].numpy().flatten() if 'bias' in state else np.zeros(num_output)
        conv = network.add_convolution(prev_layer, num_output, layer.kernel_size, W, b)
        assert conv
        conv.set_stride(layer.stride)
        conv.set_dilation(trt.infer.DimsHW(layer.dilation))
        conv.set_padding(layer.padding)
        return conv.get_output(0)

    def __activation__(self, network, prev_layer, layer):
        act = network.add_activation(prev_layer, self.ACTIVATIONS[type(layer)])
        assert act
        return act.get_output(0)

    def __pooling__(self, network, prev_layer, layer):
        k = as_tuple(layer.kernel_size)
        s = as_tuple(layer.stride)
        pool = network.add_pooling(prev_layer, self.POOLINGS[type(layer)], trt.infer.DimsHW(k))
        assert pool
        pool.set_stride(trt.infer.DimsHW(s))
        return pool.get_output(0)

    @staticmethod
    def __linear__(network, prev_layer, layer):
        num_output = layer.out_features
        state = layer.state_dict()
        W = state['weight'].numpy().flatten()
        b = state['bias'].numpy().flatten() if 'bias' in state else np.zeros(num_output)
        lin = network.add_fully_connected(prev_layer, num_output, W, b)
        assert lin
        return lin.get_output(0)

    def __sequential__(self, network, prev_layer, layer):
        net = prev_layer
        for _layer in layer.children():
            _type = type(_layer)
            if _type in self.mapping:
                net = self.mapping[_type](network, net, _layer)
                print(_type, net)
        return net

    def parse(self, module: nn.Module, input_size: tuple):
        num_channels, height, width = 3, 64, 64
        if len(input_size) == 3:
            num_channels, height, width = input_size
        elif len(input_size) == 2:
            height, width = input_size
        else:
            raise ValueError('Invalid input size, should be in form (C, H, W)')

        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
        builder = trt.infer.create_infer_builder(G_LOGGER)
        network = builder.create_network()

        net = self.__input__(network, num_channels, height, width)
        if type(module) in self.mapping:
            net = self.mapping[type(module)](network, net, module)
        else:
            raise ValueError('Invalid module subtype: {}'.format(type(module)))

        network.mark_output(net)

        builder.set_max_batch_size(1)
        builder.set_max_workspace_size(1 << 20)
        engine = builder.build_cuda_engine(network)
        network.destroy()
        builder.destroy()
        return engine
