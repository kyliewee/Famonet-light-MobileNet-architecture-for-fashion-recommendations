# Famonet-light-MobileNet-architecture-for-fashion-recommendations

## Abstract

The fashion tech space is a relatively lesser explored domain for edge applications of deep learning, even though the potential for applications is quite enormous. While deep learning CNN algorithms have been employed in multiple AI applications in the fashion tech industry, it’s still very difficult to deploy these large CNN architectures on handheld devices with limited hardware resources. Primarily, this is because CNN architectures typically consists of millions of parameters making them computationally very heavy for such small devices. We propose a novel lightweight MobileNet architecture solution, Famonet, that uses an optimal combination of separable convolutional layers, layer eliminations, max pooling, last channel reductions, and width multiplier, along with regular CNN layers. For a modest 0.7 million parameters, Famonet is able to achieve an accuracy of 94.85% on the Fashion MNIST dataset, which is comparable or better than most existing MobileNet architectures. We believe Famonet will contribute to new AI assisted use cases in the fashion retail space, where customers can get personalized fashion recommendations on the fly on handheld or mobile devices. Famonet comes at a minimal real estate of computational resources, making it, to the best of our knowledge a unique lightweight and accurate MobileNet architecture for fast fashion item recommendations on the edge.

### Dataset
We use Fashion MNIST, a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. The dataset will be automatically downloaded if it doesn't exist in your local.

### Train the model
This model was trained using Metal Performance Shaders (MPS) on a Mac. 
In the folder name 'models', there are 6 different versions of the model. We need to change one of the .py file to 'famonet' to train.
```python
python train.py -model famonet
```

### Use the tensorboard
```python
$ pip install tensorboard
$ tensorboard --logdir='runs' --port=6006 --host='localhost'
```

### Test the model
```python
python test.py -model famonet  -weightpath <path of the weight under the folder 'checkpoint'>
```
