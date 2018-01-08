from torchvision.models.alexnet import alexnet

from converter.pytorch import ModuleParser


parser = ModuleParser()
net = alexnet(pretrained=True)
engine = parser.parse(net, (3, 224, 224))

