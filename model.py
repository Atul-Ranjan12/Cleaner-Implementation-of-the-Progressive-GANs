import tensorflow as tf
from helper_functions import stage_of_resolution, num_filters, NCHW_to_NHWC, NHWC_to_NCHW, resolution_of_stage
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization


def standard_conv(filters, kernel_size, padding="same", strides=1, activation=tf.nn.leaky_relu):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  strides=strides,
                                  activation=activation, )


## The Class which scales the weights for equalized learning rate
class WeightedSum(tf.keras.layers.Add):
    """""
        Fades in the outputs from two separate paths:
            Path A: Downscale the inputs
            Path B: Through the Conv Layer
        Alpha is the scaling factor 
        returns: ((1 - alpha) * inputs) + (alpha + inputs))
    """""
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__()
        self.alpha = tf.keras.backend.variable(alpha, name="ws_alpha")

    def _merge_function(self, inputs):
        assert len(inputs) == 2
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


## Pixel Normalization Layer
class PixelNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def call(self, x, **kwargs):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)


# The Discriminator Block
def DiscriminatorBlock(stage, kernel_size=3, strides=1, padding="same"):
    """
    :rtype: object
    """
    discriminator_block = tf.keras.models.Sequential(
        [
            standard_conv(filters=num_filters(stage + 1),
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding),
            standard_conv(filters=num_filters(stage),
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding),
            tf.keras.layers.AveragePooling2D()
        ]
    )
    return discriminator_block


## The Minibatch Standard Deviation Layer
class MinibatchStdDevLayer(tf.keras.layers.Layer):
    def __init__(self, group_size=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size

    def call(self, inputs, **kwargs):
        x = NHWC_to_NCHW(inputs)
        # Minibatch must be divisible by (or smaller than) group_size.
        group_size = tf.minimum(self.group_size, tf.shape(x)[0])
        # [NCHW]  Input shape.
        s = x.shape
        # [GMCHW] Split minibatch into M groups of size G.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])
        # [GMCHW] Cast to FP32.
        y = tf.cast(y, tf.float32)
        # [GMCHW] Subtract mean over group.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        # [MCHW]  Calc variance over group.
        y = tf.reduce_mean(tf.square(y), axis=0)
        # [MCHW]  Calc stddev over group.
        y = tf.sqrt(y + 1e-8)
        # [M111]  Take average over fmaps and pixels.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)
        # [M111]  Cast back to original data type.
        y = tf.cast(y, x.dtype)
        # [N1HW]  Replicate over group and pixels.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])
        # [NCHW]  Append as new fmap.
        z = tf.concat([x, y], axis=1)
        ## Convert it back to NHWC
        z_converted = NCHW_to_NHWC(z)
        return z_converted


## The Discriminator class
class Discriminator(tf.keras.models.Sequential):
    def __init__(self, resolution=4, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.resolution = resolution
        self.stage = stage_of_resolution(self.resolution)
        self.in_shape = tf.TensorShape([self.resolution, self.resolution, 3])
        self.disc_block = [tf.keras.layers.InputLayer(input_shape=self.in_shape)]

        self.disc_block.append(
            MinibatchStdDevLayer()
        )
        self.disc_block.append(
            standard_conv(filters=num_filters(1), kernel_size=3)
        )
        self.disc_block.append(
            standard_conv(filters=num_filters(1), kernel_size=4, padding="valid")
        )
        self.disc_block.append(
            tf.keras.layers.Flatten()
        )
        self.disc_block.append(
            tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)
        )

    def call(self, inputs, training=None, mask=None):
        if self.stage == 0:
            #print("Def was called inside the discriminator "
            #      f"The stage is: {self.stage}")
            x = tf.nn.l2_normalize(inputs, axis=-1)
            for layer in self.disc_block:
                x = layer(x)
                # print(x.shape)
            return x
        else:
            ## Path A: downsample the image
            downsample = standard_conv(filters=512, kernel_size=(1, 1))(inputs)
            block_old = AveragePooling2D()(downsample)
            # print(f"block_old{block_old.shape}")
            ## Path B: Convolutional Layer
            features = self.disc_block[0](inputs)
            from_rgb = self.disc_block[1](features)
            # print(f"from_rgb{from_rgb.shape}")
            x = WeightedSum()([block_old, from_rgb])
            # print(f"x: {x.shape}")
            start_layer = self.stage + 3
            for i in range(start_layer, len(self.disc_block)):
                # print(f"Works {i}")
                a = self.disc_block[i]
                x = a(x)
                # print(x.shape)
            return x

    #### ISSUE: grow function not copying weights from the previous layers
    def grow(self):
        # print("Grow was called inside the discriminator "
        #      f"Discriminator now inputting images of resolution {self.resolution}x{self.resolution}")
        self.stage += 1
        self.resolution *= 2
        d1 = DiscriminatorBlock(self.stage)
        self.disc_block.insert(1, d1)


## Generator Block
def GeneratorBlock(stage, kernel_size=3, strides=2, padding="same"):
    filters = num_filters(stage)
    generator_block = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                            strides=strides, padding=padding,
                                            activation=tf.nn.leaky_relu),
            BatchNormalization(),
            standard_conv(filters=filters, kernel_size=kernel_size),
            BatchNormalization()
        ]
    )
    return generator_block


class Generator(tf.keras.models.Sequential):
    def __init__(self, latent_dim = 512, normalize_latents = True, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.stage = 0
        self.normalize_latents = normalize_latents
        self.filters = num_filters(self.stage)
        ## Initial block
        self.initial_layers = tf.keras.Sequential(
            [
                BatchNormalization(),
                tf.keras.layers.Dense(units=self.filters * 4 * 4),
                tf.keras.layers.Reshape([4, 4, self.filters]),
                BatchNormalization(),
                standard_conv(filters=self.filters,
                              kernel_size=3),
                BatchNormalization()
            ]
        )
        self.prog_block = [tf.keras.layers.InputLayer([latent_dim]), self.initial_layers]
        self.resolution: int = resolution_of_stage(self.stage)
        self.images = []

    def call(self, inputs, training=None, mask=None):
        x = tf.nn.l2_normalize(inputs, axis=-1) if self.normalize_latents else inputs
        for layer in self.prog_block:
            x = layer(x)
        input_shape = x.shape
        x = tf.keras.layers.Conv2D(3, (1, 1), activation=tf.keras.activations.relu, input_shape=input_shape)(x)
        self.images.append(x)
        new = self.images[-1]
        if self.stage == 0:
            return new
        else:
            old = self.images[-2]
            old_upscaled = UpSampling2D()(old)
            mix = WeightedSum()([new, old_upscaled])
            return mix

    def grow(self):
        self.stage += 1
        self.resolution *= 2
        # print(f"Grow was called inside the Generator: \n"
        #       f"Generator now generating images of resolution {self.resolution} x {self.resolution}")
        generator_block = GeneratorBlock(self.stage)
        self.prog_block.append(generator_block)
