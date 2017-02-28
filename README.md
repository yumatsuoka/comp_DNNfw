# tf_multi_gpu
Run neural network on multi gpu

## Requiremets
- chainer==1.21.0
- conda==4.3.8
- Keras==1.2.2
- scikit-learn==0.18.1
- tensorflow-gpu==1.0.0

## How to use

```
dataset = 'cifar10'
fm = 'tf'

cifar = cifar10.py if dataset == 'cifar10' else cifar100.py
python cifar

framework = {'tf': tf_pure_cifar.py, 'tfslim': tf_slim_learn_cifar.py,
             'chainer': chainer_cifar.py, 'keras': keras_cifar.py}
python framework[fm]
```

## Code
- pure_tf_cifar.py => run tensorflow code with "pure-tensorflow modules"
- chainer_cifar.py => run chainer code
- chainer_model.py => note neural net model
- keras_cifar.py => run chainer code
- keras_model.py => note neural net model
- tf_cifar.py => run neural net model with tensorflow contrib API
- tf_model.py => note neural net model with the tensorflow API
- cifar10.py => download cifar10 datset and make it dict
- cifar100.py => download cifar100 datset and make it dict
- ln_cifar_dataset.py => make synbolic link about cifar dataset
- run_annie.py => run some test to observe gpu trianing on some env
