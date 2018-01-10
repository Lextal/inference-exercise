import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from torchvision.models.alexnet import alexnet
from torch import nn

ACTIVATIONS = {
        nn.ReLU: trt.infer.ActivationType.RELU,
        nn.Sigmoid: trt.infer.ActivationType.SIGMOID,
        nn.Tanh: trt.infer.ActivationType.TANH
    }

POOLINGS = {
    nn.MaxPool2d: trt.infer.PoolingType.MAX,
    nn.AvgPool2d: trt.infer.PoolingType.AVERAGE
}


tensor_type = trt.infer.Tensor


def as_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def __convolution__(network, prev_layer, layer):
    num_output = layer.out_channels
    state = layer.state_dict()
    W = state['weight'].numpy().flatten()
    b = state['bias'].numpy().flatten() if 'bias' in state else np.zeros(num_output)
    conv = network.add_convolution(
        prev_layer.get_output(0) if not isinstance(prev_layer, tensor_type) else prev_layer,
        num_output, layer.kernel_size, W, b)
    assert conv
    conv.set_stride(layer.stride)
    conv.set_dilation(trt.infer.DimsHW(layer.dilation))
    conv.set_padding(layer.padding)
    return conv#.get_output(0)


def __activation__(network, prev_layer, layer):
    act = network.add_activation(
        prev_layer.get_output(0) if not isinstance(prev_layer, tensor_type) else prev_layer,
        ACTIVATIONS[type(layer)])
    assert act
    return act


def __pooling__(network, prev_layer, layer):
    k = as_tuple(layer.kernel_size)
    s = as_tuple(layer.stride)
    pool = network.add_pooling(
        prev_layer.get_output(0) if not isinstance(prev_layer, tensor_type) else prev_layer,
        POOLINGS[type(layer)], trt.infer.DimsHW(k))
    assert pool
    pool.set_stride(trt.infer.DimsHW(s))
    return pool


def __linear__(network, prev_layer, layer):
    num_output = layer.out_features
    state = layer.state_dict()
    W = state['weight'].numpy().flatten()
    b = state['bias'].numpy().flatten() if 'bias' in state else np.zeros(num_output)
    lin = network.add_fully_connected(
        prev_layer.get_output(0) if not isinstance(prev_layer, tensor_type) else prev_layer,
        num_output, W, b)
    assert lin
    return lin

def __sequential__(network, prev_layer, layer):
    net = prev_layer
    for _layer in layer.children():
        _type = type(_layer)
        if _type in mapping:
            net = mapping[_type](network, net, _layer)
    return net

mapping = {
            nn.Sequential: __sequential__,

            nn.Conv2d: __convolution__,

            nn.Linear: __linear__,

            nn.ReLU: __activation__,
            nn.Tanh: __activation__,
            nn.Sigmoid: __activation__,

            nn.AvgPool2d: __pooling__,
            nn.MaxPool2d: __pooling__
        }

model = alexnet(pretrained=True)
layers = []
for c in model.features.children():
    layers.append(c)
for c in model.classifier.children():
    layers.append(c)

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
builder = trt.infer.create_infer_builder(G_LOGGER)

network = builder.create_network()



net = network.add_input("data", trt.infer.DataType.FLOAT, (3, 224, 224))

# for module in model.children():
#     _type = type(module)
#     net = mapping[_type](network, net, module)

net = __convolution__(network, net, layers[0])
net = __activation__(network, net.get_output(0), layers[1])
net = __pooling__(network, net.get_output(0), layers[2])
net = __convolution__(network, net.get_output(0), layers[3])
net = __activation__(network, net.get_output(0), layers[4])
net = __pooling__(network, net.get_output(0), layers[5])
net = __convolution__(network, net.get_output(0), layers[6])
net = __activation__(network, net.get_output(0), layers[7])
net = __convolution__(network, net.get_output(0), layers[8])
net = __activation__(network, net.get_output(0), layers[9])
net = __convolution__(network, net.get_output(0), layers[10])
net = __activation__(network, net.get_output(0), layers[11])
net = __pooling__(network, net.get_output(0), layers[12])
net = __linear__(network, net.get_output(0), layers[14])
net = __activation__(network, net.get_output(0), layers[15])
net = __linear__(network, net.get_output(0), layers[17])
net = __activation__(network, net.get_output(0), layers[18])
net = __linear__(network, net.get_output(0), layers[19])

network.mark_output(net.get_output(0))

builder.set_max_batch_size(1)
builder.set_max_workspace_size(1 << 20)

engine = builder.build_cuda_engine(network)
network.destroy()
builder.destroy()

runtime = trt.infer.create_infer_runtime(G_LOGGER)
img = np.random.randn(3, 224, 224)
img = img.ravel()

context = engine.create_execution_context()

output = np.empty(1000, dtype=np.float32)

# alocate device memory
d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()
# transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
# execute model
context.enqueue(1, bindings, stream.handle, None)
# transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)

stream.synchronize()
print("Prediction: " + str(np.argmax(output)))
