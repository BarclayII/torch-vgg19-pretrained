
import torch as T
import torchvision as TV
import scipy.misc as SPM
import sys
import numpy as NP
import pickle

with open('imagenet_labels2') as f:
    classes = pickle.load(f)
x = SPM.imread(sys.argv[1], mode='RGB')
x = SPM.imresize(x, (224, 224))
x = NP.expand_dims(x, 0)

x = x.transpose([0, 3, 1, 2]) / 255.
norm = TV.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
x = T.autograd.Variable(norm(T.FloatTensor(x)))
vgg19 = TV.models.vgg19(pretrained=True)
y = vgg19(x).data.numpy()
print classes[y.argmax()]
