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
    def __init__(self, block_class, n=18):
        super(ResNet, self).__init__()
        w = math.sqrt(2)
        links = [('conv1', L.Convolution2D(3, 16, 3, 1, 0, w))]
        links += [('bn1', L.BatchNormalization(16))]
        for i in range(n):
            links += [('res{}'.format(len(links)), block_class(16, 16))]
        for i in range(n):
            links += [('res{}'.format(len(links)),
                       block_class(32 if i > 0 else 16, 32, 1 if i > 0 else 2))]
        for i in range(n):
            links += [('res{}'.format(len(links)),
                       block_class(64 if i > 0 else 32, 64,
                                   1 if i > 0 else 2))]
        links += [('_apool{}'.format(len(links)),
                   F.AveragePooling2D(8, 1, 0))]
        links += [('fc{}'.format(len(links)),
                   L.Linear(64, 10))]
        for link in links:
            if not link[0].startswith('_'):
                self.add_link(*link)
        self.forward = links
        self.train = True
    def __call__(self, x):
        for name, f in self.forward:
            if 'res' in name:
                x = f(x, self.train)
            else:
                x = f(x)
        return x


### AllConvNet
class AllConvNetBN(chainer.Chain):

    def __init__(self):
        super(AllConvNetBN, self).__init__(
                conv1 = L.Convolution2D(3, 96, 3, pad=1),
                conv2 = L.Convolution2D(96, 96, 3, pad=1),
                bn1 = L.BatchNormalization(96),
                # stridced conv
                conv3 = L.Convolution2D(96, 96, 3, stride=2),
                conv4 = L.Convolution2D(96, 192, 3, pad=1),
                conv5 = L.Convolution2D(192, 192, 3, pad=1),
                bn2 = L.BatchNormalization(192),
                # stridced conv
                conv6 = L.Convolution2D(192, 192, 3, stride=2),
                conv7 = L.Convolution2D(192, 192, 3, pad=1),
                conv8 = L.Convolution2D(192, 192, 1),
                conv9 = L.Convolution2D(192, 10, 1),
        )

    def __call__(self, x, t=None, train=True):
        h = F.relu(self.conv1(F.dropout(x, ratio=0.2, train=train)))
        h = F.relu(self.conv2(h))
        h = F.relu(self.bn1(self.conv3(h)))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.bn2(self.conv6(h)))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.relu(self.conv9(h))
        # global average pooling
        h = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 10))
        return h
