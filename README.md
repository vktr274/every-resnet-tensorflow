# Every ResNet in TensorFlow 2

This repository contains implementations of every ResNet model descibed in the [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) paper by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015). The models are implemented in the [`resnets.py`](./resnets.py) file. The only requirement is TensorFlow 2 or higher (tested with TensorFlow 2.11.0).

The implemented models are:

- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152

Each model is implemented as a function that follows this signature:

```py
def ResNetN(
    input_shape: Tuple[int, int, int],
    output_units: int = 1000,
    include_top: bool = True,
    after_input: Optional[Union[Sequential, Layer]] = None,
    normalize: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
    flatten: bool = False,
    dropout_rate: float = 0.0,
) -> Model
```

where N in the function name is the number of layers in the model that should be replaced with 18, 34, 50, 101, or 152. The function takes the following parameters:

- `input_shape` - Shape of the input images
- `output_units` - Number of output units used in the last layer if `include_top` is `True` (default: `1000`)
- `include_top` - Whether to include the network top after global average pooling or the flatten layer (default: `True`)
- `after_input` - Custom layers to add after the input like preprocessing layers as a Keras model of class [`tf.keras.Sequential`](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential) or as a single layer of class [`tf.keras.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) (default: `None` - no custom layers)
- `normalize` - Whether to normalize the input images to the range [0, 1] (default: `False`)
- `kernel_regularizer` - Kernel regularizer of class [`tf.keras.regularizers.Regularizer`](https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/Regularizer) or as a string (default: `None`)
- `kernel_initializer` - Kernel initializer of class [`tf.keras.initializers.Initializer`](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Initializer) or as a string (default: `"he_uniform"`)
- `flatten` - Whether to use a flatten layer instead of a global average pooling layer after the last block (default: `False` - use global average pooling)
- `dropout_rate` - Dropout rate used after global average pooling or flattening (default: `0.0`)

Models use the [Functional API](https://www.tensorflow.org/guide/keras/functional) of Keras under the hood which defines the model's structure as a directed acyclic graph of layers. The function returns a [`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model) instance that needs to be compiled and trained.

The implementation was tested by successfully creating each model and printing its summary to text files. A helper function to print model summaries to text files was used:

```py
def write_summary(model: Model, file_path: str) -> None:
    with open(file_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
```

The text files are included in the repository and are named [`resnet18.txt`](./resnet18.txt), [`resnet34.txt`](./resnet34.txt), [`resnet50.txt`](./resnet50.txt), [`resnet101.txt`](./resnet101.txt), and [`resnet152.txt`](./resnet152.txt).
