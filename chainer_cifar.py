from __future__ import print_function
import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

from chainer_model import ResNet, ResBlock, AllConvNetBN 


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--same_batch', '-s', type=bool, default=False,
                        help='if True and use multi gpu, batchsize*gpu_num')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu_num', '-gn', type=int, default=1,
                        help='a number of GPU(negative value indicates CPU)')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='main GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='allconvnet',
                        help='choose training model')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')

    args = parser.parse_args()
    print('# a number of using GPU: {}'.format(args.gpu_num))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    
    # make dump name with this experiment
    dump_dir = './result/train_log'+'_gpu_num-'+str(args.gpu_num)+"_model-"+str(args.model)+'_epoch-'+str(args.epoch)+'_batchsize-'+str(args.batchsize)+'_datset-'+str(args.dataset)

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('# Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('# Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')
    if args.model == 'resnet':
        print('# cnn_model: resnet')
        model = L.Classifier(ResNet(class_labels=class_labels))
    elif args.model == 'allconvnet':
        print('# cnn_model: AllConvNetBN')
        model = L.Classifier(AllConvNetBN(class_labels))
    else:
        raise RuntimeError('Invalid dataset choice.')

    if args.gpu >= 0 and args.gpu_num >= 1:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current

    #optimizer = chainer.optimizers.MomentumSGD(0.01)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))
    
    
    #multi gpu環境、つまりParallelUpdaterを使った並列GPU処理だとbatchsize = batchsize/gpu_num
    batchsize = args.batchsize * args.gpu_num if args.same_batch else args.batchsize
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    if args.gpu_num <= 1:
        print("# main gpu: ", args.gpu)
        model.to_gpu()  # Copy the model to the GPU
        updater = training.StandardUpdater(train_iter, optimizer,device=args.gpu)
    elif args.gpu_num >= 2: 
        _devices = {'main': args.gpu}
        print("# main gpu: ", args.gpu)
        for g_idx in range(1, args.gpu_num):
            _devices[str(g_idx)] = g_idx
        print("# using gpus: ", _devices)
        updater = training.ParallelUpdater(
            train_iter, 
            optimizer, 
            devices=_devices,
        )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=dump_dir)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(TestModeEvaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('alpha', 0.5),
                   trigger=(20, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        print('Resume from a snapshot')
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()
