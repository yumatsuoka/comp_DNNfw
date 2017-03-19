# deep learning with multi gpu on some framework
Run neural network on multi gpu

## Requiremets
- chainer==1.21.0
- conda3==4.3.8
- Keras==1.2.2
- scikit-learn==0.18.1
- tensorflow-gpu==1.0.0

## How to use

```
dataset = 'cifar10'
fm = 'tf'

# download dataset
cifar = cifar10.py if dataset == 'cifar10' else cifar100.py
python cifar

# run training
framework = {'tf': tf_pure_cifar.py, 'tfslim': tf_slim_learn_cifar.py,
             'chainer': chainer_cifar.py, 'keras': keras_cifar.py}
python framework[fm]
```

## Code
- chainer_cifar.py => run chainer code
- chainer_model.py => note neural net model
- cifar10.py => download cifar10 datset and make it dict
- cifar100.py => download cifar100 datset and make it dict
- keras_cifar.py => run chainer code
- keras_make_parallel.py => note multi gpu processing
- keras_model.py => note neural net model
- ln_cifar_dataset.py => make synbolic link about cifar dataset
- run_experiment.py => run some test to observe gpu trianing on some env
- tf_pure_cifar.py => run neural net model with tensorflow contrib API
- tf_pure_datafeeder.py => note datafeeder(this is future work)
- tf_pure_model.py => note neural net model with the tensorflow API
- tf_pure_trainer.py => note trainer on single and multi gpu 
