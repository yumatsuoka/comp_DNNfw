import chainer
import chainer.functions as F
import chainer.links as L
import math

class ResBlock(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1, ksize=1):
        w = math.sqrt(2)
        super(ResBlock, self).__init__(
            conv1=L.Convolution2D(n_in, n_out, 3, stride, 1, w),
            bn1=L.BatchNormalization(n_out),
            conv2=L.Convolution2D(n_out, n_out, 3, 1, 1, w),
            bn2=L.BatchNormalization(n_out),
        )
    def __call__(self, x, train):
        h = F.relu(self.bn1(self.conv1(x), test=not train))
        h = self.bn2(self.conv2(h), test=not train)
        if x.data.shape != h.data.shape:
            xp = chainer.cuda.get_array_module(x.data)
            n, c, hh, ww = x.data.shape
            pad_c = h.data.shape[1] - c
            p = xp.zeros((n, pad_c, hh, ww), dtype=xp.float32)
            p = chainer.Variable(p, volatile='auto')
            #p = chainer.Variable(p, volatile=not train)
            x = F.concat((p, x))
            if x.data.shape[2:] != h.data.shape[2:]:
                x = F.average_pooling_2d(x, 1, 2)
        return F.relu(h + x)

class ResNet(chainer.Chain):
    def __init__(self, block_class=ResBlock, n=18, class_labels=10):
        super(ResNet, self).__init__()
        w = math.sqrt(2)
        links = [('conv1', L.Convolution2D(3, 96, 3, 1, 0, w))]
        links += [('bn1', L.BatchNormalization(96))]
        for i in range(n):
            links += [('res{}'.format(len(links)), block_class(96, 96))]
        for i in range(n):
            links += [('res{}'.format(len(links)),
                       block_class(128 if i > 0 else 96, 128, 1 if i > 0 else 2))]
        for i in range(n):
            links += [('res{}'.format(len(links)),
                       block_class(192 if i > 0 else 128, 192,
                                   1 if i > 0 else 2))]
        links += [('_apool{}'.format(len(links)),
                   F.AveragePooling2D(8, 1, 0))]
        links += [('fc{}'.format(len(links)),
                   L.Linear(192, class_labels))]
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
        self.train = True
        print(links)
    def __call__(self, x):
        for name, f in self.forward:
            if 'res' in name:
                x = f(x, self.train)
            else:
                x = f(x)
        return x


### AllConvNet
class AllConvNetBN(chainer.Chain):

    def __init__(self, class_labels=10):
        super(AllConvNetBN, self).__init__(
                conv1 = L.Convolution2D(3, 96, 3, pad=1),
                bn1 = L.BatchNormalization(96),
                conv2 = L.Convolution2D(96, 96, 3, pad=1),
                bn2 = L.BatchNormalization(96),
                # stridced conv
                conv3 = L.Convolution2D(96, 96, 3, stride=2),
                bn3 = L.BatchNormalization(96),
                conv4 = L.Convolution2D(96, 192, 3, pad=1),
                bn4 = L.BatchNormalization(192),
                conv5 = L.Convolution2D(192, 192, 3, pad=1),
                bn5 = L.BatchNormalization(192),
                # stridced conv
                conv6 = L.Convolution2D(192, 192, 3, stride=2),
                bn6 = L.BatchNormalization(192),
                conv7 = L.Convolution2D(192, 192, 3, pad=1),
                bn7 = L.BatchNormalization(192),
                conv8 = L.Convolution2D(192, 192, 1),
                bn8 = L.BatchNormalization(192),
                conv9 = L.Convolution2D(192, class_labels, 1),
                bn9 = L.BatchNormalization(10),
        )
        self.train = True
        self.class_labels = class_labels

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(F.dropout(x, ratio=0.2, train=self.train))))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        h = F.relu(self.bn4(self.conv4(h)))
        h = F.relu(self.bn5(self.conv5(h)))
        h = F.relu(self.bn6(self.conv6(h)))
        h = F.relu(self.bn7(self.conv7(h)))
        h = F.relu(self.bn8(self.conv8(h)))
        h = F.relu(self.bn9(self.conv9(h)))
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], self.class_labels))
        return h
