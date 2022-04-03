### Importing the dependencies and scripts
import tensorflow as tf
from helper_functions import num_filters, stage_of_resolution, resolution_of_stage, NCHW_to_NHWC, NHWC_to_NCHW
from typing import List


## Convolution layer that scales the weights for an equalized learning rate.
class WSConv2D(tf.keras.layers.Layer):
    """""
    Scales the outputs of the Convolutional Layer so that the weights are scaled for an equalized
    learning rate. In Convolutional Layers, the learning rate for some weights may be more than required
    for other weights. This layer fixes it.
    """""
    def __init__(self, in_channels, filters, kernel_size=3, strides=1, padding="same", gain=2, *args, **kwargs):
        super(WSConv2D, self).__init__(*args, **kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
        )
        self.scale = (gain/(in_channels * (kernel_size**2))) ** 0.5

    def call(self, inputs, **kwargs):
        return self.conv(inputs * self.scale)


## Convolution Layer
def standard_conv(filters, kernel_size, padding="same", strides=1, activation=tf.nn.leaky_relu):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  strides=strides,
                                  activation=activation)


## Returns a Convolutional layer with 3 Filters
def rgb_conv(kernel_size, padding="same", strides=1, filters=3):
    return tf.keras.layers.Conv2D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  strides=strides)


## The Upscale class
class Upscale2D(tf.keras.layers.Layer):
    def __init__(self, factor: int = 2, *args, **kwargs):
        self.factor = factor
        super().__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        if self.factor == 1:
            return inputs
        x = NHWC_to_NCHW(inputs)
        s = x.shape
        x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
        x = tf.tile(x, [1, 1, 1, self.factor, 1, self.factor])
        x = tf.reshape(x, [-1, s[1], s[2] * self.factor, s[3] * self.factor])
        x = NCHW_to_NHWC(x)
        return x

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return tf.TensorShape([*input_shape[:-2], *(d.value * 2 for d in input_shape[-2:])])


## Pixel Normalization Layer
class PixelNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def call(self, x, **kwargs):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)


## Generator Block
def GeneratorBlock(stage, kernel_size=3, strides=2, padding="same"):
    filters = num_filters(stage)
    generator_block = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                            strides=strides, padding=padding,
                                            activation=tf.nn.leaky_relu),
            PixelNorm(),
            WSConv2D(in_channels=filters, filters=filters, kernel_size=kernel_size),
            PixelNorm()
        ]
    )
    return generator_block


### TODO: 1) Add Mixing Factor to the Generator as described in the paper.
### TODO: 2) Add the RGB Layer to the Generator
## Building the generator
class Generator(tf.keras.models.Sequential):
    def __init__(self, latent_dim=512, normalize_latents=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = 0
        self.normalize_latents = normalize_latents
        self.filters = num_filters(self.stage)
        ### Initial block taking 1x1 to 4x4
        self.initial_layers = tf.keras.Sequential(
            [
                PixelNorm(),
                tf.keras.layers.Dense(units=self.filters * 4 * 4),
                tf.keras.layers.Reshape([4, 4, self.filters]),
                PixelNorm(),
                WSConv2D(in_channels=self.filters,
                         filters=self.filters,
                         kernel_size=3),
                PixelNorm()
            ]
        )
        ## Initial RGB layer
        self.initial_rgb = rgb_conv(kernel_size=1)
        self.prog_block = [tf.keras.layers.InputLayer([latent_dim]), self.initial_layers]
        self.resolution: int = resolution_of_stage(self.stage)

    def call(self, inputs, training=None, mask=None):
        print("def call was called inside the Generator Class")
        x = tf.nn.l2_normalize(inputs, axis=-1) if self.normalize_latents else inputs
        print(f"x after l2 normalize: {x.shape}")
        for layer in self.prog_block:
            x = layer(x)
        input_shape = x.shape
        x = tf.keras.layers.Conv2D(3, (1, 1), activation=tf.keras.activations.relu, input_shape=input_shape)(x)
        return x

    def grow(self):
        self.stage += 1
        self.resolution *= 2
        print(f"Grow was called inside the Generator: \n"
              f"Generator now generating images of resolution {self.resolution} x {self.resolution}")
        generator_block = GeneratorBlock(self.stage)
        self.prog_block.append(generator_block)
        print(f"Prog blocks: {self.prog_block}")

# Building the generator
# class Generator(tf.keras.models.Sequential):
#     def __init__(self, latent_dim=512, normalize_latents=True, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.stage = 0
#         self.normalize_latents = normalize_latents
#         self.filters = num_filters(self.stage)
#         ### Initial block taking 1x1 to 4x4
#         self.initial_layers = tf.keras.Sequential(
#             [
#                 PixelNorm(),
#                 tf.keras.layers.Dense(units=self.filters * 4 * 4),
#                 tf.keras.layers.Reshape([4, 4, self.filters]),
#                 PixelNorm(),
#                 WSConv2D(in_channels=self.filters,
#                          filters=self.filters,
#                          kernel_size=3),
#                 PixelNorm()
#             ]
#         )
#         ## Initial RGB layer
#         self.initial_rgb = rgb_conv(kernel_size=1)
#         self.prog_block = [tf.keras.layers.InputLayer([latent_dim]), self.initial_layers]
#         self.resolution: int = resolution_of_stage(self.stage)
#
#     @staticmethod
#     def fade_in(upscaled, generated, alpha = 0.5):
#         return tf.nn.tanh(alpha * generated + (1 - alpha) * upscaled)
#
#     def call(self, inputs, training=None, mask=None):
#         print("def call was called inside the Generator Class")
#         if self.stage == 0:
#             x = tf.nn.l2_normalize(inputs, axis=-1) if self.normalize_latents else inputs
#             for layer in self.prog_block:
#                 x = layer(x)
#             input_shape = x.shape
#             final_out = tf.keras.layers.Conv2D(3, (1, 1), activation=tf.keras.activations.relu, input_shape=input_shape)(x)
#         else:
#             # Path A: Upscale -> RGB
#             x = inputs
#             x = Upscale2D()(x)
#             print(f"Upscaled shape: {x.shape}")
#             input_shape = x.shape
#             out1 = tf.keras.layers.Conv2D(3, (1, 1), activation=tf.keras.activations.relu, input_shape=input_shape)(x)
#             # Path B: Pass it through the progressive block -> RGB
#             y = tf.nn.l2_normalize(inputs, axis=-1) if self.normalize_latents else inputs
#             for layer in self.prog_block:
#                 y = layer(y)
#             input_shape = y.shape
#             out2 = tf.keras.layers.Conv2D(3, (1, 1), activation=tf.keras.activations.relu, input_shape=input_shape)(y)
#             final_out = self.fade_in(out1, out2)
#         return final_out
#
#     def grow(self):
#         self.stage += 1
#         self.resolution *= 2
#         print(f"Grow was called inside the Generator: \n"
#               f"Generator now generating images of resolution {self.resolution} x {self.resolution}")
#         generator_block = GeneratorBlock(self.stage)
#         self.prog_block.append(generator_block)
#         print(f"Prog blocks: {self.prog_block}")


# class Generator(tf.keras.models.Sequential):
#     def __init__(self, latent_dim=512, normalize_latents=True, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.stage = 0
#         self.normalize_latents = normalize_latents
#         self.filters = num_filters(self.stage)
#         ### Initial block taking 1x1 to 4x4
#         self.initial_layers = tf.keras.Sequential(
#             [
#                 PixelNorm(),
#                 tf.keras.layers.Dense(units=self.filters * 4 * 4),
#                 tf.keras.layers.Reshape([4, 4, self.filters]),
#                 PixelNorm(),
#                 WSConv2D(in_channels=self.filters,
#                          filters=self.filters,
#                          kernel_size=3),
#                 PixelNorm()
#             ]
#         )
#         ## Initial RGB layer
#         self.initial_rgb = rgb_conv(kernel_size=1)
#         self.prog_block = [tf.keras.layers.InputLayer([latent_dim]), self.initial_layers]
#         self.resolution: int = resolution_of_stage(self.stage)
#
#     @staticmethod
#     def fade_in(upscaled, generated, alpha = 0.5, ):
#         return tf.nn.tanh(alpha * generated + (1 - alpha) * upscaled)
#
#     def call(self, inputs, training=None, mask=None):
#         print("def call was called inside the Generator Class")
#         x = tf.nn.l2_normalize(inputs, axis=-1) if self.normalize_latents else inputs
#         for layer in self.prog_block:
#             x = layer(x)
#         input_shape = x.shape
#         if self.stage == 0:
#             x = tf.keras.layers.Conv2D(3, (1, 1), activation=tf.keras.activations.relu, input_shape=input_shape)(x)
#             return x
#         else:
#             for step in range(self.stage):
#                 upscale_x = Upscale2D()(x)
#                 x = self.prog_block[step](upscaled)
#             final_upscaled = tf.keras.layers.Conv2D(3, (1, 1), activation=tf.keras.activations.relu, input_shape=input_shape)(x)
#             return self.fade_in(alpha, final_upscaled, final_out)
#
#     def grow(self):
#         self.stage += 1
#         self.resolution *= 2
#         print(f"Grow was called inside the Generator: \n"
#               f"Generator now generating images of resolution {self.resolution} x {self.resolution}")
#         generator_block = GeneratorBlock(self.stage)
#         self.prog_block.append(generator_block)
#         print(self.prog_block)


# The Discriminator block
def DiscriminatorBlock(stage, kernel_size=3, strides=1, padding = "same"):
    discriminator_block = tf.keras.models.Sequential(
        [
            WSConv2D(in_channels=num_filters(stage+1),
                     filters=num_filters(stage + 1),
                     kernel_size=kernel_size,
                     strides=strides,
                     padding=padding),
            WSConv2D(in_channels=num_filters(stage),
                     filters=num_filters(stage),
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


# ### TODO: Make the Discriminator Block
# ### TODO: 1) Implement the mixing factor for the discriminator
class Discriminator(tf.keras.models.Sequential):
    def __init__(self, resolution: int = 4, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.resolution = resolution
        self.layer_list: List[tf.keras.layers.Layer]
        self.stage = stage_of_resolution(self.resolution)
        self.in_shape = tf.TensorShape([self.resolution, self.resolution, 3])
        self.disc_block = [tf.keras.layers.InputLayer(input_shape=self.in_shape)]

        for i in range(self.stage, 0, -1):
            print(self.stage)
            d1 = DiscriminatorBlock(i)
            self.disc_block.append(d1)

        ## Adding the Minibatch Standard Deviation Layer
        self.disc_block.append(
            MinibatchStdDevLayer()
        )
        self.disc_block.append(
            WSConv2D(in_channels=num_filters(1)+1, filters=num_filters(1), kernel_size=3)
        )
        self.disc_block.append(
            WSConv2D(in_channels=num_filters(1)+1, filters=num_filters(1), kernel_size=4, padding="valid")
        )
        self.disc_block.append(
            tf.keras.layers.Flatten()
        )
        self.disc_block.append(
            tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear)
        )
        print(f"Disc Block is: {self.disc_block}")

    def call(self, inputs, training=None, mask=None):
        print("def call was called inside the discriminator")
        x = tf.nn.l2_normalize(inputs, axis=-1)
        for layer in self.disc_block:
            x = layer(x)
            print(x.shape)
        return x

    def grow(self):
        print("Grow was called inside the discriminator "
              f"Discriminator now inputting images of resolution {self.resolution}x{self.resolution}")
        new = Discriminator(resolution=self.resolution*2)
        print(f"Resolution after growing {new.resolution}")
        current_layers = {
            l.name: l for l in self.layers
        }
        for layer in new.layers:
            if layer.name in current_layers:
                print(f"layer '{layer.name}' is common.")
                layer.set_weights(current_layers[layer.name].get_weights())
        return new


### Testing Sequences (For generator):
# noise = tf.random.normal([1, 1, 1, 128])
# gen_1 = Generator()
# out = gen_1(noise)
# print(out)
# print(gen_1.resolution)
# for _ in range(8):
#     gen_1.grow()
#     out_1 = gen_1(noise)
#     print(out_1.shape)


### Testing Sequences for Discriminator
# noise = tf.random.normal([1, 4, 4, 3])
# disc_1 = Discriminator(resolution=4)
# print(disc_1)
# y = disc_1(noise)
# print(y.shape)
# noise_2 = tf.random.normal([1, 8, 8, 3])
# disc_2 = disc_1.grow()
# print(f"Resolution of Discriminator after growing: {disc_2.resolution}")
# y2 = disc_2(noise_2)
# print(f"Output shape{y2.shape}")

### Testing sequences for the generator and the discriminator
# noise = tf.random.normal([1, 512])
# gen = Generator()
# out = gen(noise)
# print(f"output shape: {out.shape}")
# print(f"Resolution:{gen.resolution}")
# disc = Discriminator()
# y = disc(out)
# print(f"Discriminator output shape: {y.shape}")
# print(f"Discriminator resolution: {disc.resolution}")
# for layer in disc.layers:
#     print(layer)
#
#
# gen.grow()
# out2 = gen(noise)
# disc1 = disc.grow()
# print("Discriminator has grown")
# print(f"Discriminator stage: {disc1.stage}")
# print(f"Discriminator resolution: {disc1.resolution}")
# y1 = disc1(out2)
# for layer in disc1.layers:
#     print(layer)


### Testing sequence for the Minibatch Standard Deviation:
# noise = tf.random.normal([1, 512])
# gen = Generator()
# out = gen(noise)
# print(f"output shape: {out.shape}")
# print(f"Resolution of the Generator:{gen.resolution}")
# minibatch_std = MinibatchStdDevLayer()(out)
# print(f"Output shape of Minibatch Standard Deviation layer {minibatch_std.shape}")


### Testing sequence for upscale layer
# noise = tf.random.normal([1, 4, 4, 3])
# x = Upscale2D()(noise)
# print(f"Noise shape: {noise.shape}")
# print(f"Upscaled noise shape: {x.shape}")
