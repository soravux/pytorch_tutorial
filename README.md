Welcome to our PyTorch tutorial!

The goal of this tutorial is to give a quick overview of PyTorch to machine learning researchers. The target audience is people already accustomed with basic neural network theory and with some other neural networks like Keras+Tensorflow, Theano, Caffe and the like. We cover the basic PyTorch features that allows tinkering, tweaking and understanding alongside inference and training.

This repository is a companion for a ~1-2 aaaaa long presentation. Two datasets are used, MNIST (vanilla from `torchvision`) and the [Kaggle's Dogs vs Cats Redux](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition) dataset, with the folder hierarchy changed to fit [`torchvision`'s ImageFolder](https://github.com/pytorch/vision) hierarchy.

The proposed architectures are definitely *not* optimal for the given tasks, they are only presented for teaching purposes.

I highly recommend the very good [PyTorch examples of Justin Johnson](https://github.com/jcjohnson/pytorch-examples) to get an overview of all the rest offered in PyTorch, like custom layers, custom autograd functions, the optimizers, etc.

Summary
-------

Goals:

* Get you started with hands-on reusable examples
* Get you interested and hint you to the right places
* Examples are targeted for research on deep learning

Not (specifically) covered:

* (Multi-)GPU acceleration (We’ll just see the basics)
* Custom layers
* Advanced stuff


Requirements
------------

All the examples were developed for PyTorch 0.1 and 0.2 on Python 3.5 and 3.6 .

For the last examples, the [cats vs dogs redux]() dataset is needed.

* [Example 1](https://github.com/soravux/pytorch_tutorial/blob/master/example1.py): Pytorch's official MNIST example with slight modifications and annotations to explain during the presentation (data loading, network architecture and train/eval loop).
* [Example 2 (gradient)](https://github.com/soravux/pytorch_tutorial/blob/master/example2_gradient.py): How to set probes inside the network to obtain the gradient before the first FC layer as a Numpy array. Also shows how to modify the gradient during training.
* [Example 2 (adversarial example)](https://github.com/soravux/pytorch_tutorial/blob/master/example2_adv_example.py): Get the gradient of the input. Subtracting this gradient to the input would generate an adversarial example.
* [Example 3 (checkpointing)](https://github.com/soravux/pytorch_tutorial/blob/master/example3.py): Example 1 with checkpointing.
* [Example 4 (dynamic graph)](https://github.com/soravux/pytorch_tutorial/blob/master/example4.py): Simple example of dynamic graph execution, one major feature difference with other frameworks.
* [Example 5 (custom dataset)](https://github.com/soravux/pytorch_tutorial/blob/master/example5.py): Custom architecture derived from example 1 to work on the cats and dogs dataset. Obtains 63% accuracy in a single epoch (12 minutes on CPU), and 73% for 2 epochs (25 minutes on CPU).
* [Example 6 (transfer learning)](https://github.com/soravux/pytorch_tutorial/blob/master/example6.py): Finetuning pretrained resnet-50 on the cats and dogs dataset. 95% accuracy in a single epoch, roughly 30 minutes training.
* [Example 6 (second example)](https://github.com/soravux/pytorch_tutorial/blob/master/example6_squeezenet.py): Finetuning pretrained squeezenet 1.1 on the cats and dogs dataset. 93% accuracy in a single epoch (13 minutes on CPU).
* [Example 6 (gradients)](https://github.com/soravux/pytorch_tutorial/blob/master/example6_gradient.py): How to set a backward hook to get gradients and foward hooks to get features on a pretrained network.
* [Example 6 (feature extractor)](https://github.com/soravux/pytorch_tutorial/blob/master/example6_features.py): Resnet-50 feature extractor on images loaded using PIL.
