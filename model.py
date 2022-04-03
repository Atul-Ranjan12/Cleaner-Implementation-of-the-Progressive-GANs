### Importing the dependencies and scripts
import tensorflow as tf
from helper_functions import num_filters, stage_of_resolution, resolution_of_stage
from typing import List


## TODO: Add a class to scale the outputs from the Convolution layer Here


## TODO: Implement the Minibatch Standard Deviation Layer here


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
            standard_conv(filters=filters, kernel_size=kernel_size),
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
                standard_conv(filters=self.filters,
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
        print(self.prog_block)


# The Discriminator block
def DiscriminatorBlock(stage, kernel_size = 3, strides=1, padding = "same"):
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


# class DiscriminatorBlock(tf.keras.models.Sequential):
#     def __init__(self, resolution: int, kernel_size=3, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.stage = stage_of_resolution(resolution)
#         self.add(standard_conv(filters=num_filters(self.stage + 1),
#                                kernel_size=kernel_size))
#         self.add(standard_conv(filters=num_filters(self.stage),
#                                kernel_size=kernel_size))
#         self.add(tf.keras.layers.AveragePooling2D())
#
#
# class Discriminator(tf.keras.models.Sequential):
#     def __init__(self, resolution: int = None, *args, **kwargs):
#         super(Discriminator, self).__init__(*args, **kwargs)
#         self.resolution = resolution
#         self.layer_list: List[tf.keras.layers.Layer]
#         if self.resolution is not None:
#             self.build(tf.TensorShape([self.resolution, self.resolution, 3]))
#
#     def build(self, input_shape=None):
#         self.resolution = input_shape[1]
#         self.stage = stage_of_resolution(self.resolution)
#
#         self.add(tf.keras.layers.InputLayer([self.resolution, self.resolution, 3]))
#         self.add(tf.keras.layers.Conv2D(filters=num_filters(self.stage),
#                                         kernel_size=1,
#                                         strides=1))
#
#         for i in range(self.stage, 0, -1):
#             self.add(DiscriminatorBlock(resolution_of_stage(i)))
#
#         self.add(standard_conv(filters=num_filters(1), kernel_size=3))
#         self.add(standard_conv(filters=num_filters(1), kernel_size=4, padding="valid"))
#         self.add(tf.keras.layers.Flatten())
#         self.add( tf.keras.layers.Dense(units=1, activation=tf.keras.activations.linear))
#
#     def call(self, inputs, training=None, mask=None):
#         print("def call was called inside the discriminator")
#         x = tf.nn.l2_normalize(inputs, axis=-1)
#         for layer in self.layers:
#             x = layer(x)
#             print(x.shape)
#         return x
#
#     def grow(self):
#         print("Grow was called inside the discriminator "
#               f"Discriminator now inputting images of resolution {self.resolution}x{self.resolution}")
#         new_resolution = self.resolution * 2
#         print(f"Actual resolution after growing: {new_resolution}")
#         new = Discriminator(resolution=new_resolution)
#         print(f"Resolution after growing {new.resolution}")
#         current_layers = {
#             l.name: l for l in self.layers
#         }
#         for layer in new.layers:
#             if layer.name in current_layers:
#                 print(f"layer '{layer.name}' is common.")
#                 layer.set_weights(current_layers[layer.name].get_weights())
#         return new


# ### TODO: Make the Discriminator Block
# ### TODO: 1) Implement the mixing factor for the discriminator
# ### TODO: 2) Implement the Minibatch Standard Deviation Layer in the discriminator
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
# print(out.shape)
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
# print(f"Resolution the generator is now outputting: {gen.resolution}")
# print(f"Stage of the resolution of the generator: {gen.stage}")
# disc1 = disc.grow()
# print("Discriminator has grown")
# print(f"Discriminator stage: {disc1.stage}")
# print(f"Discriminator resolution: {disc1.resolution}")
# for layer in disc1.layers:
#     print(layer)