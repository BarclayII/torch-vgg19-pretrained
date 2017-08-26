
import torch as T
import torchvision as TV
import scipy.misc as SPM
import sys
import numpy as NP
import pickle

imgs = []
with open('imagenet_labels2') as f:
    classes = pickle.load(f)

for filename in sys.argv[1:]:
    x = SPM.imread(filename, mode='RGB')
    x = SPM.imresize(x, (224, 224))
    x = NP.expand_dims(x, 0)

    x = x.transpose([0, 3, 1, 2]) / 255.
    imgs.append(x)
x = NP.concatenate(imgs)
norm = TV.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#x = T.autograd.Variable(T.stack([norm(T.FloatTensor(_)) for _ in x]))
x = T.autograd.Variable(norm(T.FloatTensor(x)))
vgg19 = TV.models.vgg19(pretrained=True)
vgg19.eval()
y = vgg19(x).data.numpy()
for idx in y.argmax(axis=1):
    print classes[idx]
