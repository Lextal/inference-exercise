import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from torchvision.models.alexnet import alexnet

model = alexnet(pretrained=True)
layers = []
for c in model.features.children():
    layers.append(c)
for c in model.classifier.children():
    layers.append(c)

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
builder = trt.infer.create_infer_builder(G_LOGGER)

network = builder.create_network()

data = network.add_input("data", trt.infer.DataType.FLOAT, (3, 224, 224))
assert data

weights = layers[0].state_dict()
conv1_W = weights['weight'].numpy().flatten()
conv1_b = weights['bias'].numpy().flatten()
net = network.add_convolution(data, 64, (11, 11), conv1_W, conv1_b)
net.set_stride((4, 4))
net.set_padding((2, 2))
net.set_name('conv1')
print(net)
assert net

net = network.add_activation(net.get_output(0), trt.infer.ActivationType.RELU)
print(net)
assert net

net = network.add_pooling(net.get_output(0), trt.infer.PoolingType.MAX, (3, 3))
print(net)
assert (net)
net.set_stride((2, 2))

weights = layers[3].state_dict()
conv1_W = weights['weight'].numpy().flatten()
conv1_b = weights['bias'].numpy().flatten()
net = network.add_convolution(net.get_output(0), 192, (5, 5), conv1_W, conv1_b)
net.set_name('conv2')
print(net)
assert net
net.set_padding((2, 2))

net = network.add_activation(net.get_output(0), trt.infer.ActivationType.RELU)
print(net)
assert net

net = network.add_pooling(net.get_output(0), trt.infer.PoolingType.MAX, (3, 3))
print(net)
assert (net)
net.set_stride((2, 2))

weights = layers[6].state_dict()
conv1_W = weights['weight'].numpy().flatten()
conv1_b = weights['bias'].numpy().flatten()
net = network.add_convolution(net.get_output(0), 384, (3, 3), conv1_W, conv1_b)
net.set_name('conv3')
print(net)
assert net
net.set_padding((1, 1))

net = network.add_activation(net.get_output(0), trt.infer.ActivationType.RELU)
print(net)
assert net

weights = layers[8].state_dict()
conv1_W = weights['weight'].numpy().flatten()
conv1_b = weights['bias'].numpy().flatten()
net = network.add_convolution(net.get_output(0), 256, (3, 3), conv1_W, conv1_b)
net.set_name('conv4')
print(net)
assert net
net.set_padding((1, 1))

net = network.add_activation(net.get_output(0), trt.infer.ActivationType.RELU)
print(net)
assert net

weights = layers[10].state_dict()
conv1_W = weights['weight'].numpy().flatten()
conv1_b = weights['bias'].numpy().flatten()
net = network.add_convolution(net.get_output(0), 256, (3, 3), conv1_W, conv1_b)
net.set_name('conv5')
print(net)
assert net
net.set_padding((1, 1))

net = network.add_activation(net.get_output(0), trt.infer.ActivationType.RELU)
print(net)
assert net

net = network.add_pooling(net.get_output(0), trt.infer.PoolingType.MAX, (3, 3))
print(net)
assert (net)
net.set_stride((2, 2))

weights = layers[14].state_dict()
fc2_w = weights['weight'].cpu().numpy().reshape(-1)
fc2_b = weights['bias'].cpu().numpy().reshape(-1)
net = network.add_fully_connected(net.get_output(0), 4096, fc2_w, fc2_b)
net.set_name('fc1')
print(net)
assert (net)

net = network.add_activation(net.get_output(0), trt.infer.ActivationType.RELU)
print(net)
assert net

weights = layers[17].state_dict()
fc2_w = weights['weight'].cpu().numpy().reshape(-1)
fc2_b = weights['bias'].cpu().numpy().reshape(-1)
net = network.add_fully_connected(net.get_output(0), 4096, fc2_w, fc2_b)
net.set_name('fc2')
print(net)
assert (net)

net = network.add_activation(net.get_output(0), trt.infer.ActivationType.RELU)
print(net)
assert net

weights = layers[19].state_dict()
fc2_w = weights['weight'].cpu().numpy().reshape(-1)
fc2_b = weights['bias'].cpu().numpy().reshape(-1)
net = network.add_fully_connected(net.get_output(0), 1000, fc2_w, fc2_b)
net.set_name('fc3')
print(net)
assert (net)

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
# syncronize threads
stream.synchronize()
print("Prediction: " + str(np.argmax(output)))
