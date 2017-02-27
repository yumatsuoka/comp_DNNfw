import os

model = 'allconvnet'
gpu = 0
epoch = 100

for sb in [False, True]:
    for gn in [1]:
        os.system("python chainer_cifar.py -e {} -m {} -g {} -gn {} -s {}".format(epoch, model, gpu, gn, sb))
