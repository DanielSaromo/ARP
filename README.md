# Auto-Rotating Perceptrons (ARP)

This repository contains the Keras implementation of the Auto-Rotating Perceptrons (Saromo, Villota, and Villanueva) for dense layers of artificial neural networks. These neural units were presented in [this paper](https://arxiv.org/abs/1910.02483) with an [oral exposition](https://slideslive.com/38922594/autorotating-perceptrons) at the [LXAI workshop](https://nips.cc/Conferences/2019/Schedule?showEvent=15988) at [NeurIPS 2019](https://neurips.cc/Conferences/2019).

The ARP library was developed by [Daniel Saromo](https://www.danielsaromo.xyz/) and [Matias Valdenegro-Toro](https://mvaldenegro.github.io/). This repository contains implementations that are not present in the [LXAI @ NeurIPS 2019 paper](https://arxiv.org/abs/1910.02483).

## What is an Auto-Rotating Perceptron? 
The ARP are a generalization of the perceptron unit that aims to avoid the vanishing gradient problem by making the activation function's input near zero, without altering the inference structure learned by the perceptron.

| Classic perceptron | Auto-Rotating Perceptron | 
| --- | --- |
| <img src="https://www.danielsaromo.xyz/assets/img/neuronas_classic.svg" height="200"> | <img src="https://www.danielsaromo.xyz/assets/img/neuronas_ARP.svg" height="200"> | 

[comment]: <> (render en svg y embed en HTML: https://stackoverflow.com/questions/11256433/how-to-show-math-equations-in-general-githubs-markdownnot-githubs-blog)
[comment]: <> (https://stackoverflow.com/questions/47344571/how-to-draw-checkbox-or-tick-mark-in-github-markdown-table)

Hence, a classic perceptron becomes the particular case of an ARP with `rho=1`.

### Basic principle: The dynamic region

We define the *dynamic region* as the symmetric numerical range (w.r.t. `0`) from where we would like the pre-activation values to come from in order to avoid node saturation. Recall that, in order to avoid the vanishing gradient problem (VGP), we do not want the derivative of the activation function to take tiny values. For the ARP, the dynamic region goes from `-L` to `+L`.

For example, in the unipolar sigmoid activation function (logistic curve) shown below, we would like it to receive values from `-4` to `4`. For inputs whose absolute values are higher than `4`, the derivative of the activation is too low. Hence, the `L` value could be `4`. The resulting *dynamic region* projected on the derivative curve is depicted as a gray shade.

<img src="https://www.danielsaromo.xyz/assets/img/sigmoid_and_deriv.jpg" height="300">

### What is `L`?

`L` is the hyperparameter that defines the limits of the desired symmetric *dynamic region*.

### How do I choose `L`?

You need to analyze the activation function and its derivative. For the dynamic region, there is a trade-off. For a bigger `L`, you accept more non-linearity for the activation function, but at the same time, you get more saturation.

Below you have the suggested values for `L`, according to the activation function of the neuron:

| Activation function | `L` | 
|:-:|:-:|
| tanh           | 2         |
| sigmoid        | 4         |
| arctan         | 7         |

In the figure below, you can see that for inputs whose absolute values are higher than the values from the table, the derivative of the activation functions is very small.

<img src="https://www.danielsaromo.xyz/assets/img/saturated_activations.jpg" height="300">

### What about `xQ`?

In the [original ARP paper](https://arxiv.org/abs/1910.02483), you needed to set this value manually. Currently, by default, `xQ` is automatically calculated using `L`. However, the ARP library supports a custom selection of the `xQ` value.

A deeper explanation can be found in the journal version of the ARP paper (in preparation).

### ARP Vs. Classic perceptrons

As shown in the example notebook (`examples/example_CIFAR10_Keras.ipynb`) that compares ARP and classic perceptrons, the ARP can lead to a faster convergence and lower loss values.

Furthermore, there is an application of the ARP to calibrate a wearable sensor where the test loss was reduced by a factor of 15 when changing from classic perceptrons to ARP. You can check the paper [here](https://www.danielsaromo.xyz/publications/#SSC_with_ARP).

In machine learning, the advantages of using one technique or another are problem-dependant. We encourage you to apply ARP in your research and discover its potential.


## Keras implementation

### Instalation

The ARP library is available on the [Python Package Index](https://pypi.org/project/arpkeras/ "ARP Keras page on PyPI").
To install the library, first install `pip` and then use the following command:

```python
pip install arpkeras
```

You may need to update the `pip` manager. You can use:
```python
python -m pip install –upgrade pip
```

### Import

```python
from ARPkeras import AutoRotDense
```

### Creating an ARP model

The `AutoRotDense` class implementation inherits from the Keras `Dense` class. Hence, you can use it as a typical Keras `Dense` layer, but adding the following arguments:

- `xmin_lim`: The lower limit of the values that will enter the neuron. For example, if we scale our input data to the range `0` to `1`, and we choose `tanh` as the activation function (which goes from `-1` to `+1`), then the lowest input value will be `xmin_lim=-1`.
- `xmax_lim`: The upper limit of the values that will enter the neuron. Analogous to `xmin_lim`.
- `L` : The limit of the desired symmetrical *dynamic region*. This value is the **only hyperparameter** needed for the Auto-Rotating layers. The two variables described above depend on the activation function you choose and your data preprocessing.

This is an example of use for the unipolar `'sigmoid'` activation (whose output goes from `0` to `+1`) with data scaled to the range `0` to `1`:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ARPkeras import AutoRotDense

xmin_lim = 0   # min( 0,0)
xmax_lim = 1   # max(+1,1)
L = 4

model = Sequential()
model.add(Dense(20), input_shape=(123,))
model.add(AutoRotDense(10, xmin_lim=xmin_lim, xmax_lim=xmax_lim, L=L, activation='sigmoid'))
#By default the `AutoRot` flag of the Auto-Rotating layers is True.

model.summary()
```

This is another example, when using the `'tanh'` activation (whose output goes from `-1` to `+1`) with data scaled to the range `0` to `1`:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ARPkeras import AutoRotDense

xmin_lim = -1   # min(-1,0)
xmax_lim = +1   # max(+1,1)
L = 7

model = Sequential()
model.add(AutoRotDense(20, input_shape=(123,), xmin_lim=xmin_lim, xmax_lim=xmax_lim, L=L, activation='tanh'))
model.add(AutoRotDense(10, xmin_lim, xmax_lim, L, activation='tanh'))
#By default the `AutoRot` flag of the Auto-Rotating layers is True.

model.summary()
```

### Beyond `Dense` layers

Is that all for the ARP? No! The journal version of the ARP paper is being finished, with the support of Dr. Edwin Villanueva and Dr. Matias Valdenegro-Toro. There, the Auto-Rotating concept was extrapolated to other layer types, creating the **Auto-Rotating Neural Networks**.

These are the Keras layers implemented with the Auto-Rotating operation (**Tip**: Just add `AutoRot` before the layer name):

| Keras Original Layer    | Auto-Rotating Implementation | 
|:-:|:-:|
| `Dense`               | `AutoRotDense`              |
| `SimpleRNN`           | `AutoRotSimpleRNN`          |
| `LSTM`                | `AutoRotLSTM`               |
| `GRU`                 | `AutoRotGRU`                |
| `Conv1D`              | `AutoRotConv1D`             |
| `Conv2D`              | `AutoRotConv2D`             |
| `Conv3D`              | `AutoRotConv3D`             |
| `Conv2DTranspose`     | `AutoRotConv2DTranspose`    |
| `Conv3DTranspose`     | `AutoRotConv3DTranspose`    |
| `SeparableConv`       | `AutoRotSeparableConv`      |
| `SeparableConv1D`     | `AutoRotSeparableConv1D`    |
| `SeparableConv2D`     | `AutoRotSeparableConv2D`    |
| `DepthwiseConv2D`     | `AutoRotDepthwiseConv2D`    |

Coming soon :sunglasses:: **Auto-Rotating Neural Networks** (Saromo, Villanueva, and Valdenegro-Toro).

## Citation

[comment]: <> (We hope this code and our paper can help researchers, scientists, and engineers to improve the use and design of Auto-Rotating models that have potentially exciting applications in deep learning.)

This code is free to use for research purposes, and if used or modified in any way, please consider citing:

```
@article{saromo2019arp,
  title={{A}uto-{R}otating {P}erceptrons},
  author={Saromo, Daniel and Villota, Elizabeth and Villanueva, Edwin},
  journal={LatinX in AI Workshop at NeurIPS 2019 (arXiv:1910.02483)},
  year={2019}
}
```

Other inquiries: daniel.saromo@pucp.pe