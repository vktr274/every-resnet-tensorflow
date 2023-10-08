# ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152
# based on the 'Deep Residual Learning for Image Recognition' paper
# by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun
# https://arxiv.org/pdf/1512.03385.pdf

from typing import Optional, Tuple, Union
from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import (
    Dropout,
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Dense,
    GlobalAveragePooling2D,
    Add,
    BatchNormalization,
    ReLU,
    Softmax,
    Flatten,
    Layer,
)


def ResidualBlockLarge(
    x_in,
    filters: Tuple[int, int, int],
    s: int = 1,
    reduce: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
):
    """
    Create a ResNet block with 3 layers

    :param x_in: input tensor
    :param filters: number of filters in each layer
    :param s: stride used when reducing the input tensor
    :param reduce: whether to reduce the input tensor
    :param kernel_regularizer: kernel regularizer
    :param kernel_initializer: the kernel initializer to use

    :return: output tensor
    """
    filters1, filters2, filters3 = filters

    y_out = Conv2D(
        filters1,
        kernel_size=(1, 1),
        strides=(s, s),
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(x_in)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(
        filters2,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(y_out)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(
        filters3,
        kernel_size=(1, 1),
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(y_out)
    y_out = BatchNormalization()(y_out)

    if reduce:
        x_in = Conv2D(
            filters3,
            kernel_size=(1, 1),
            strides=(s, s),
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
        )(x_in)
        x_in = BatchNormalization()(x_in)

    y_out = Add()([y_out, x_in])

    return ReLU()(y_out)


def ResidualBlockSmall(
    x_in,
    filters: Tuple[int, int],
    s: int = 1,
    reduce: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
):
    """
    Create a ResNet block with 2 layers

    :param x_in: input tensor
    :param filters: number of filters in each layer
    :param s: stride used when reducing the input tensor
    :param reduce: whether to reduce the input tensor
    :param kernel_regularizer: kernel regularizer
    :param kernel_initializer: the kernel initializer to use

    :return: output tensor
    """
    filters1, filters2 = filters

    y_out = Conv2D(
        filters1,
        kernel_size=(3, 3),
        strides=(s, s),
        padding="same",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(x_in)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = Conv2D(
        filters2,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(y_out)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    if reduce:
        x_in = Conv2D(
            filters2,
            kernel_size=(1, 1),
            strides=(s, s),
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
        )(x_in)
        x_in = BatchNormalization()(x_in)

    y_out = Add()([y_out, x_in])

    return ReLU()(y_out)


def ResNet(
    input_shape: Tuple[int, int, int],
    block_sizes: Tuple[int, int, int, int],
    net_size: str,
    output_units: int = 1000,
    include_top: bool = True,
    after_input: Optional[Union[Sequential, Layer]] = None,
    normalize: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
    flatten: bool = False,
    dropout_rate: float = 0.0,
) -> Model:
    """
    Create one of ResNet-18, ResNet-34, ResNet-50, ResNet-101, and ResNet-152

    :param input_shape: Shape of the input images.
    :param block_sizes: Number of layers in each block.
    :param net_size: Size of ResNet 'small' for ResNet-18 and ResNet-34, 'large' for ResNet-50, ResNet-101, and ResNet-152.
    :param output_units: NNumber of output units used in the last layer if include_top is True (default: 1000).
    :param include_top: Whether to include the network top after global average pooling or the flatten layer (default: True).
    :param after_input: Custom layers to add after the input like preprocessing layers as a Keras model of class
    tf.keras.Sequential or as a single layer of class tf.keras.layers.Layer (default: None - no custom layers).
    :param normalize: Whether to normalize the input images to the range [0, 1] (default: False).
    :param kernel_regularizer: Kernel regularizer of class tf.keras.regularizers.Regularizer or as a string (default: None).
    :param kernel_initializer: Kernel initializer of class tf.keras.initializers.Initializer or as a string (default: "he_uniform").
    :param flatten: Whether to use a flatten layer instead of a global average pooling layer after the last block
    (default: False - use global average pooling).
    :param dropout_rate: Dropout rate used after global average pooling or flattening (default: 0.0).

    :return: ResNet model.
    """
    if net_size not in ("small", "large"):
        raise ValueError("Invalid net_size value. Must be 'small' or 'large'.")

    x_in = Input(shape=input_shape)
    y_out = x_in

    if normalize:
        y_out = Rescaling(scale=1.0 / 255)(y_out)

    if after_input is not None:
        y_out = after_input(y_out)

    y_out = Conv2D(
        64,
        kernel_size=(7, 7),
        strides=(2, 2),
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
    )(y_out)
    y_out = BatchNormalization()(y_out)
    y_out = ReLU()(y_out)

    y_out = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(y_out)

    block1, block2, block3, block4 = block_sizes

    for layer in range(block1):
        y_out = (
            ResidualBlockLarge(
                y_out,
                (64, 64, 256),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
            )
            if net_size == "large"
            else ResidualBlockSmall(
                y_out,
                (64, 64),
                s=1,
                reduce=False,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
            )
        )

    for layer in range(block2):
        y_out = (
            ResidualBlockLarge(
                y_out,
                (128, 128, 512),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
            )
            if net_size == "large"
            else ResidualBlockSmall(
                y_out,
                (128, 128),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
            )
        )

    for layer in range(block3):
        y_out = (
            ResidualBlockLarge(
                y_out,
                (256, 256, 1024),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
            )
            if net_size == "large"
            else ResidualBlockSmall(
                y_out,
                (256, 256),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
            )
        )

    for layer in range(block4):
        y_out = (
            ResidualBlockLarge(
                y_out,
                (512, 512, 2048),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
            )
            if net_size == "large"
            else ResidualBlockSmall(
                y_out,
                (512, 512),
                s=2 if layer == 0 else 1,
                reduce=layer == 0,
                kernel_regularizer=kernel_regularizer,
                kernel_initializer=kernel_initializer,
            )
        )

    y_out = Flatten()(y_out) if flatten else GlobalAveragePooling2D()(y_out)
    if dropout_rate > 0.0:
        y_out = Dropout(dropout_rate)(y_out)

    if include_top:
        y_out = Dense(
            output_units,
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
        )(y_out)
        y_out = Softmax()(y_out)

    return Model(inputs=x_in, outputs=y_out)


def ResNet18(
    input_shape: Tuple[int, int, int],
    output_units: int = 1000,
    include_top: bool = True,
    after_input: Optional[Union[Sequential, Layer]] = None,
    normalize: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
    flatten: bool = False,
    dropout_rate: float = 0.0,
) -> Model:
    """
    Create a ResNet-18 model.

    :param input_shape: Shape of the input images.
    :param output_units: NNumber of output units used in the last layer if include_top is True (default: 1000).
    :param include_top: Whether to include the network top after global average pooling or the flatten layer (default: True).
    :param after_input: Custom layers to add after the input like preprocessing layers as a Keras model of class
    tf.keras.Sequential or as a single layer of class tf.keras.layers.Layer (default: None - no custom layers).
    :param normalize: Whether to normalize the input images to the range [0, 1] (default: False).
    :param kernel_regularizer: Kernel regularizer of class tf.keras.regularizers.Regularizer or as a string (default: None).
    :param kernel_initializer: Kernel initializer of class tf.keras.initializers.Initializer or as a string (default: "he_uniform").
    :param flatten: Whether to use a flatten layer instead of a global average pooling layer after the last block
    (default: False - use global average pooling).
    :param dropout_rate: Dropout rate used after global average pooling or flattening (default: 0.0).

    :return: ResNet-18 model.
    """
    return ResNet(
        input_shape,
        (2, 2, 2, 2),
        "small",
        output_units=output_units,
        include_top=include_top,
        after_input=after_input,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        flatten=flatten,
        dropout_rate=dropout_rate,
    )


def ResNet34(
    input_shape: Tuple[int, int, int],
    output_units: int = 1000,
    include_top: bool = True,
    after_input: Optional[Union[Sequential, Layer]] = None,
    normalize: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
    flatten: bool = False,
    dropout_rate: float = 0.0,
) -> Model:
    """
    Create a ResNet-34 model.

    :param input_shape: Shape of the input images.
    :param output_units: NNumber of output units used in the last layer if include_top is True (default: 1000).
    :param include_top: Whether to include the network top after global average pooling or the flatten layer (default: True).
    :param after_input: Custom layers to add after the input like preprocessing layers as a Keras model of class
    tf.keras.Sequential or as a single layer of class tf.keras.layers.Layer (default: None - no custom layers).
    :param normalize: Whether to normalize the input images to the range [0, 1] (default: False).
    :param kernel_regularizer: Kernel regularizer of class tf.keras.regularizers.Regularizer or as a string (default: None).
    :param kernel_initializer: Kernel initializer of class tf.keras.initializers.Initializer or as a string (default: "he_uniform").
    :param flatten: Whether to use a flatten layer instead of a global average pooling layer after the last block
    (default: False - use global average pooling).
    :param dropout_rate: Dropout rate used after global average pooling or flattening (default: 0.0).

    :return: ResNet-34 model.
    """
    return ResNet(
        input_shape,
        (3, 4, 6, 3),
        "small",
        output_units=output_units,
        include_top=include_top,
        after_input=after_input,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        flatten=flatten,
        dropout_rate=dropout_rate,
    )


def ResNet50(
    input_shape: Tuple[int, int, int],
    output_units: int = 1000,
    include_top: bool = True,
    after_input: Optional[Union[Sequential, Layer]] = None,
    normalize: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
    flatten: bool = False,
    dropout_rate: float = 0.0,
) -> Model:
    """
    Create a ResNet-50 model.

    :param input_shape: Shape of the input images.
    :param output_units: NNumber of output units used in the last layer if include_top is True (default: 1000).
    :param include_top: Whether to include the network top after global average pooling or the flatten layer (default: True).
    :param after_input: Custom layers to add after the input like preprocessing layers as a Keras model of class
    tf.keras.Sequential or as a single layer of class tf.keras.layers.Layer (default: None - no custom layers).
    :param normalize: Whether to normalize the input images to the range [0, 1] (default: False).
    :param kernel_regularizer: Kernel regularizer of class tf.keras.regularizers.Regularizer or as a string (default: None).
    :param kernel_initializer: Kernel initializer of class tf.keras.initializers.Initializer or as a string (default: "he_uniform").
    :param flatten: Whether to use a flatten layer instead of a global average pooling layer after the last block
    (default: False - use global average pooling).
    :param dropout_rate: Dropout rate used after global average pooling or flattening (default: 0.0).

    :return: ResNet-50 model.
    """
    return ResNet(
        input_shape,
        (3, 4, 6, 3),
        "large",
        output_units=output_units,
        include_top=include_top,
        after_input=after_input,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        flatten=flatten,
        dropout_rate=dropout_rate,
    )


def ResNet101(
    input_shape: Tuple[int, int, int],
    output_units: int = 1000,
    include_top: bool = True,
    after_input: Optional[Union[Sequential, Layer]] = None,
    normalize: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
    flatten: bool = False,
    dropout_rate: float = 0.0,
) -> Model:
    """
    Create a ResNet-101 model.

    :param input_shape: Shape of the input images.
    :param output_units: NNumber of output units used in the last layer if include_top is True (default: 1000).
    :param include_top: Whether to include the network top after global average pooling or the flatten layer (default: True).
    :param after_input: Custom layers to add after the input like preprocessing layers as a Keras model of class
    tf.keras.Sequential or as a single layer of class tf.keras.layers.Layer (default: None - no custom layers).
    :param normalize: Whether to normalize the input images to the range [0, 1] (default: False).
    :param kernel_regularizer: Kernel regularizer of class tf.keras.regularizers.Regularizer or as a string (default: None).
    :param kernel_initializer: Kernel initializer of class tf.keras.initializers.Initializer or as a string (default: "he_uniform").
    :param flatten: Whether to use a flatten layer instead of a global average pooling layer after the last block
    (default: False - use global average pooling).
    :param dropout_rate: Dropout rate used after global average pooling or flattening (default: 0.0).

    :return: ResNet-101 model.
    """
    return ResNet(
        input_shape,
        (3, 4, 23, 3),
        "large",
        output_units=output_units,
        include_top=include_top,
        after_input=after_input,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        flatten=flatten,
        dropout_rate=dropout_rate,
    )


def ResNet152(
    input_shape: Tuple[int, int, int],
    output_units: int = 1000,
    include_top: bool = True,
    after_input: Optional[Union[Sequential, Layer]] = None,
    normalize: bool = False,
    kernel_regularizer: Optional[Union[Regularizer, str]] = None,
    kernel_initializer: Union[Initializer, str] = "he_uniform",
    flatten: bool = False,
    dropout_rate: float = 0.0,
) -> Model:
    """
    Create a ResNet-152 model.

    :param input_shape: Shape of the input images.
    :param output_units: NNumber of output units used in the last layer if include_top is True (default: 1000).
    :param include_top: Whether to include the network top after global average pooling or the flatten layer (default: True).
    :param after_input: Custom layers to add after the input like preprocessing layers as a Keras model of class
    tf.keras.Sequential or as a single layer of class tf.keras.layers.Layer (default: None - no custom layers).
    :param normalize: Whether to normalize the input images to the range [0, 1] (default: False).
    :param kernel_regularizer: Kernel regularizer of class tf.keras.regularizers.Regularizer or as a string (default: None).
    :param kernel_initializer: Kernel initializer of class tf.keras.initializers.Initializer or as a string (default: "he_uniform").
    :param flatten: Whether to use a flatten layer instead of a global average pooling layer after the last block
    (default: False - use global average pooling).
    :param dropout_rate: Dropout rate used after global average pooling or flattening (default: 0.0).

    :return: ResNet-152 model.
    """
    return ResNet(
        input_shape,
        (3, 8, 36, 3),
        "large",
        output_units=output_units,
        include_top=include_top,
        after_input=after_input,
        normalize=normalize,
        kernel_regularizer=kernel_regularizer,
        kernel_initializer=kernel_initializer,
        flatten=flatten,
        dropout_rate=dropout_rate,
    )


def write_summary(model: Model, file_path: str) -> None:
    """Write a summary of the model to a text file.

    :param model: The model to summarize.
    :param file_path: The path to the text file to write.
    """
    with open(file_path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


if __name__ == "__main__":
    # Summarize models to test implementation.
    input_shape = (224, 224, 3)
    normalize = True

    model = ResNet18(input_shape, normalize=normalize)
    write_summary(model, "resnet18.txt")

    model = ResNet34(input_shape, normalize=normalize)
    write_summary(model, "resnet34.txt")

    model = ResNet50(input_shape, normalize=normalize)
    write_summary(model, "resnet50.txt")

    model = ResNet101(input_shape, normalize=normalize)
    write_summary(model, "resnet101.txt")

    model = ResNet152(input_shape, normalize=normalize)
    write_summary(model, "resnet152.txt")
