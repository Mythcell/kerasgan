"""
Custom Keras models for training GANs and CGANs.

Copyright (C) 2022 Mythcell

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Last updated: 26 Aug, 2022
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython import display

# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------

class GAN(keras.Model):
    """A model for training a generative adversarial network"""

    def __init__(self, generator, discriminator, latent_dim):
        """
        Instantiate a generative adversarial network.

        Args:
            generator (keras.Model): The generator. Takes a latent vector as
                input and outputs an image.
            discriminator (keras.Model): The discriminator. Takes an image as input
                and outputs a value.
            latent_dim (int): Size of the latent vector. This must be equal to the
                generator's input.
        """
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    @property
    def metrics(self):
        return [self.g_loss_metric, self.d_loss_metric]

    def compile(self, generator_optimizer, discriminator_optimizer):
        """
        Compile the generative adversarial network. Recommended optimizer:
        tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.05)

        Args:
            generator_optimizer: Optimizer to use for the generator
            discriminator_optimizer: Optimizer to use for the discriminator
        """
        super(GAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')

    def discriminator_loss(self, real_output, fake_output):
        # Note the use of label smoothing
        real_loss = self.cross_entropy(
            tf.random.uniform(tf.shape(real_output), minval=0.9, maxval=1), real_output)
        fake_loss = self.cross_entropy(
            tf.random.uniform(tf.shape(fake_output), minval=0, maxval=0.1), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        # first sample from the latent space
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dsc_tape:
            # generate fake images
            generated_images = self.generator(noise, training=True)

            # determine outputs of the discriminator for both real and fake images
            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # determine losses
            gen_loss = self.generator_loss(fake_output)
            dsc_loss = self.discriminator_loss(real_output, fake_output)

        # calculate gradients
        gradients_gen = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        gradients_dsc = dsc_tape.gradient(
            dsc_loss, self.discriminator.trainable_variables)

        # apply gradients to update weights
        self.generator_optimizer.apply_gradients(
            zip(gradients_gen, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_dsc, self.discriminator.trainable_variables))

        # update metrics
        self.g_loss_metric.update_state(gen_loss)
        self.d_loss_metric.update_state(dsc_loss)
        return {'g_loss':self.g_loss_metric.result(),
            'd_loss':self.d_loss_metric.result()}


class CGAN(GAN):
    """A model for training a conditional generative adversarial network"""

    def __init__(self, generator, discriminator, latent_dim, num_classes):
        """
        A conditional GAN model.

        Args:
            generator (keras.Model): The generator. Must accept two inputs:
                - latent vector, of shape (latent_dim,), and
                - label, of shape (1,)
                The output of the generator must be an image
            discriminator (keras.Model): The discriminator. Must accept two inputs:
                - image, of shape (image_size, image_size, image_channels)
                - label, of shape (1,)
                The final layer of the discriminator should be a Dense() layer
                with no activation.
            latent_dim (int): Length of the latent vector
            num_classes (int): Number of distinct classes. All labels should be
                strictly in the range [0, 1, ..., num_classes - 1]
        """
        super().__init__(generator, discriminator, latent_dim)
        self.num_classes = num_classes
    
    @tf.function
    def train_step(self, data):
        # unpack images and labels
        images, labels = data
        # determine number of samples present in the current batch (important!)
        batch_size = tf.shape(images)[0]

        # first generate random latent vectors
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dsc_tape:
            # generate fake images
            generated_images = self.generator([noise, labels], training=True)

            # determine outputs of the discriminator for both real and fake images
            real_output = self.discriminator([images, labels], training=True)
            fake_output = self.discriminator([generated_images, labels], training=True)

            # determine losses
            gen_loss = self.generator_loss(fake_output)
            dsc_loss = self.discriminator_loss(real_output, fake_output)

        # calculate gradients
        gradients_gen = gen_tape.gradient(
            gen_loss,self.generator.trainable_variables)
        gradients_dsc = dsc_tape.gradient(
            dsc_loss,self.discriminator.trainable_variables)

        # apply gradients to update weights
        self.generator_optimizer.apply_gradients(
            zip(gradients_gen, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_dsc, self.discriminator.trainable_variables))

        # update metrics and return losses
        self.g_loss_metric.update_state(gen_loss)
        self.d_loss_metric.update_state(dsc_loss)
        return {'g_loss':self.g_loss_metric.result(),
            'd_loss':self.d_loss_metric.result()}


# ----------------------------------------------------------------------------
# CALLBACKS
# ----------------------------------------------------------------------------


class GANSnapshot(keras.callbacks.Callback):
    """Generates images with a given seed at the end of every epoch"""

    def __init__(self, seed=None, num_images=32, freq=10, figscale=1, dpi=100,
                 cmap='binary_r', show_snapshots=True, save_snapshots=False,
                 save_freq=10, save_dir='gan_snapshots', save_prefix='',
                 epoch_fmt = '05d'):
        """
        Generates images with the provided seed (or a random seed if no seed is
        provided). Intended to be used to help inspect and track the progress of
        the generator as it is trained. This snapshot is inteded to be used with
        GAN() models, for CGAN() models use CGANSnapshot.

        Args:
            seed (tf.Tensor): Latent vectors to generate images with.
                These should be generated with tf.random.normal(). If None,
                a seed is automatically generated with length equal to num_images.
            num_images (int): Number of images to generate in each snapshot. This
                is only used to generate a random seed, and is ignored if a seed
                is provided. Defaults to 32.
            freq (int): How often to generate a snapshot. Default is every 10 epochs.
            figscale (float): Base figure scale multiplier. Defaults to 1.
            dpi (float): Base resolution. Defaults to 100.
            cmap (str): Colormap to use for mono images (is ignored for RGB images).
            show_snapshots (bool): Whether to display the snapshots with plt.show.
                Recommend setting to False if using a non-interactive workflow.
            save_snapshots (bool): Whether to save images to file. Defaults to False.
                Images are saved as .png files and named by epoch.
            save_dir (str): Directory to save images to. Default is gan_snapshots.
            save_freq (int): How often to save images (in units of epochs).
                Defaults to saving an image every 10 epochs.
            save_prefix (str): Prefix to prepend to the filename of each image.
                e.g. 'model1_' --> files saved as model1_0000.png, model1_0001.png
            epoch_fmt (str): String format to use for the epoch. Defaults to '05d'.
        """
        self.seed = seed
        self.num_images = num_images
        self.freq = freq
        self.figscale = figscale
        self.dpi = dpi
        self.cmap = cmap
        self.show_snapshots = show_snapshots
        self.save_snapshots = save_snapshots
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.save_prefix = save_prefix
        self.epoch_fmt = epoch_fmt

    def on_train_begin(self, logs=None):
        # create a random seed if none was initially provided
        if self.seed is None:
            self.seed = tf.random.normal(shape=(self.num_images,self.model.latent_dim))

        if self.save_snapshots:
            try:
                os.mkdir(self.save_dir)
            except FileExistsError:
                pass

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.freq != 0:
            return
        if self.show_snapshots:
            display.clear_output(wait=True)

        gen_im = self.model.generator.predict(self.seed, verbose=0)
        gen_im = np.uint8((gen_im + 1)*127.5)
        is_bw = gen_im.shape[-1] == 1

        fig_height, fig_width = get_figure_dimensions(len(gen_im))
        fig = plt.figure(figsize=(self.figscale*fig_width,
            self.figscale*fig_height),dpi=self.dpi)
        fig.subplots_adjust(wspace=0.01,hspace=0.01)

        for i,gim in enumerate(gen_im):
            fig.add_subplot(fig_height, fig_width, i+1)
            if is_bw:
                plt.imshow(gim[...,0],cmap=self.cmap)
            else:
                plt.imshow(gim)
            plt.axis('off')
        if self.show_snapshots:
            plt.show()
            
        if self.save_snapshots and epoch % self.save_freq == 0:
            fig.savefig(os.path.join(self.save_dir,f'{epoch:{self.epoch_fmt}}.png'),
                format='png',facecolor='black',bbox_inches='tight')
        fig.clear()
        plt.close(fig)


class CGANSnapshot(keras.callbacks.Callback):
    """Generate images with the provided seed and labels."""

    def __init__(self, seed=None, labels=None, num_images=32, freq=10, figscale=1,
                 dpi=100, cmap='binary_r', show_snapshots=True, save_snapshots=False,
                 save_dir='cgan_snapshots', save_freq=10, save_prefix='',
                 epoch_fmt='05d'):
        """
        Generates images with the provided seed and labels (or with random seeds
        and random labels if none are provided). Intended to be used to help inspect
        and track the progress of the generator as it is trained. If one of seed and
        labels is none, the other will be randomly generated. This snapshot is intended
        to be used with CGAN() models. For GAN() models use GANSnapshot.

        Args:
            seed (tf.Tensor): Latent vectors to generate images with.
                These should be generated with tf.random.normal(). If None,
                a seed is automatically generated with length equal to num_images.
            labels (np.array): List of integer labels, e.g. [0, 1, 2, 3, ...].
                Values should be strictly less than the total number of classes.
                If None, labels are automatically generated with values from 0 up to
                CGAN.num_classes - 1.
            num_images (int): Number of images to generate. Ignored if seeds or labels
                is provided. Defaults to 32.
            freq (int): How often to generate a snapshot. Default is every 10 epochs.
            figscale (float): Base figure scale multiplier. Defaults to 1.
            dpi (float): Base resolution. Defaults to 100.
            cmap (str): Colormap to use for mono images (is ignored for RGB images).
            show_snapshots (bool): Whether to display the snapshots with plt.show.
                Recommend setting to False if using a non-interactive workflow.
            save_snapshots (bool): Whether to save images to file. Defaults to False.
                Images are saved as .png files and named by epoch.
            save_dir (str): Directory to save images to. Default is cgan_snapshots
            save_freq (int): How often to save images in units of epochs.
                Defaults to saving an image every 10 epochs.
            save_prefix (str): Prefix to prepend to the filename of each image.
                e.g. 'model1_' --> files saved as model1_0000.png, model1_0001.png
            epoch_fmt (str): String format to use for the epoch. Defaults to '05d'.
        """
        self.seed = seed
        self.labels = labels
        self.num_images = num_images
        self.freq = freq
        self.figscale = figscale
        self.dpi = dpi
        self.cmap = cmap
        self.show_snapshots = show_snapshots
        self.save_snapshots = save_snapshots
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_prefix = save_prefix
        self.epoch_fmt = epoch_fmt

    def on_train_begin(self, logs=None):
        # first generate some labels if none were provided
        if self.labels is None:
            if self.seed is None:
                self.labels = tf.random.uniform(
                    shape=[self.num_images], minval=0, maxval=self.model.num_classes
                )
            else:
                self.labels = tf.random.uniform(
                    shape=[len(self.seed)], minval=0, maxval=self.model.num_classes
                )

        # generate a seed if none was initially provided
        if self.seed is None:
            self.seed = tf.random.normal(
                shape=(len(self.labels), self.model.latent_dim))

        # make save directory if it does not yet exist
        if self.save_snapshots:
            try:
                os.mkdir(self.save_dir)
            except FileExistsError:
                pass

    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.freq != 0:
            return
        if self.show_snapshots:
            display.clear_output(wait=True)

        gen_im = self.model.generator.predict([self.seed, self.labels],verbose=0)
        gen_im = np.uint8((gen_im + 1)*127.5) # convert to [0, 255]
        is_bw = gen_im.shape[-1] == 1

        fig_height, fig_width = get_figure_dimensions(len(gen_im))

        fig = plt.figure(figsize=(self.figscale*fig_width,
            self.figscale*fig_height),dpi=self.dpi)
        fig.subplots_adjust(wspace=0.05,hspace=0.05)

        for i,gim in enumerate(gen_im):
            fig.add_subplot(fig_height, fig_width, i+1)
            if is_bw:
                plt.imshow(gim[...,0],cmap=self.cmap)
            else:
                plt.imshow(gim)
            plt.axis('off')
        if self.show_snapshots:
            plt.show()

        if self.save_snapshots and epoch % self.save_freq == 0:
            fig.savefig(os.path.join(
                f'{self.save_dir}',f'{self.save_prefix}{epoch:{self.epoch_fmt}}.png'),
                format='png',facecolor='white',bbox_inches='tight')
        fig.clear()
        plt.close(fig)


class CGANClassSnapshot(keras.callbacks.Callback):

    def __init__(self, seed=None, examples_per_class=3, columns_indicate='classes',
                 freq=10, figscale=1, dpi=100, cmap='binary_r', show_snapshots=True,
                 save_snapshots=False, save_dir='cgan_class_snapshots', save_freq=10,
                 save_prefix='',epoch_fmt='05d'):
        """
        Generates example images for every class, either with the provided seed
        or with a random seed (if not provided). By default, the figure will contain
        [num_classes] columns and [examples_per_class] rows (can be swapped).
        Intended to be used to help inspect and track the progress of the generator
        as it is trained. This snapshot should only be used with CGANs.

        Args:
            seed (tf.Tensor): Latent vectors to generate images with.
                These should be generated with tf.random.normal(). If None,
                a seed is automatically generated with length equal to
                CGAN.num_classes * examples_per_class.
            examples_per_class (int): Number of distinct images
                to generate for each class.
            columns_indicate (str): One of 'classes' or 'examples' (default: 'classes').
                If set to 'classes', the figure will be arranged so that each column
                shows a different class, and each row shows a different example.
                If set to 'examples', the above roles are reversed.
            freq (int): How often to generate a snapshot. Default is every 10 epochs.
            figscale (float): Base figure scale multiplier. Defaults to 1.
            dpi (float): Base resolution. Defaults to 100.
            cmap (str): Colormap to use for mono images (is ignored for RGB images).
            show_snapshots (bool): Whether to display the snapshots with plt.show.
                Recommend setting to False if using a non-interactive workflow.
            save_snapshots (bool): Whether to save images to file. Defaults to False.
                Images are saved as .png files and named by epoch.
            save_dir (str): Directory to save images to.
                Default is 'cgan_class_snapshots'
            save_freq (int): How often to save images in units of epochs.
                Defaults to saving an image every 10 epochs.
            save_prefix (str): Prefix to prepend to the filename of each image.
                e.g. 'model1_' --> files saved as model1_0000.png, model1_0001.png
            epoch_fmt (str): String format to use for the epoch. Defaults to '05d'.
        """
        self.seed = seed
        self.labels = None
        self.examples_per_class = examples_per_class
        self.columns_indicate = columns_indicate
        self.freq = freq
        self.figscale = figscale
        self.dpi = dpi
        self.cmap = cmap
        self.show_snapshots = show_snapshots
        self.save_snapshots = save_snapshots
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_prefix = save_prefix
        self.epoch_fmt = epoch_fmt

    def on_train_begin(self, logs=None):

        if self.columns_indicate == 'classes':
            labels = list(range(self.model.num_classes)) * self.examples_per_class
        elif self.columns_indicate == 'examples':
            labels = []
            for i in range(self.model.num_classes):
                labels.extend([i]*self.examples_per_class)
        else:
            raise ValueError('CGANClassSnapshot: '
                + "columns_indicate must be one of 'classes' or 'examples'")
        self.labels = np.array(labels)

        if self.seed is None:
            self.seed = tf.random.normal(shape=(
                self.examples_per_class*self.model.num_classes,self.model.latent_dim))

        if self.save_snapshots:
            try:
                os.mkdir(self.save_dir)
            except FileExistsError:
                pass
    
    def on_epoch_end(self, epoch, logs=None):

        if epoch % self.freq != 0:
            return
        if self.show_snapshots:
            display.clear_output(wait=True)

        gen_im = self.model.generator.predict([self.seed,self.labels],verbose=0)
        gen_im = np.uint8((gen_im + 1)*127.5)
        is_bw = gen_im.shape[-1] == 1

        if self.columns_indicate == 'classes':
            fig_width, fig_height = self.model.num_classes, self.examples_per_class
        else:
            fig_width, fig_height = self.examples_per_class, self.model.num_classes

        fig = plt.figure(figsize=(self.figscale*fig_width,
            self.figscale*fig_height),dpi=self.dpi)
        fig.subplots_adjust(wspace=0.05,hspace=0.05)

        for i,gim in enumerate(gen_im):
            fig.add_subplot(fig_height, fig_width, i+1)
            if is_bw:
                plt.imshow(gim[...,0],cmap=self.cmap)
            else:
                plt.imshow(gim)
            plt.axis('off')
        if self.show_snapshots:
            plt.show()

        if self.save_snapshots and epoch % self.save_freq == 0:
            fig.savefig(os.path.join(
                f'{self.save_dir}',f'{self.save_prefix}{epoch:{self.epoch_fmt}}.png'),
                format='png',facecolor='white',bbox_inches='tight')
        fig.clear()
        plt.close(fig)


class GANCheckpoint(keras.callbacks.Callback):
    """Periodically saves the generator and discriminator"""

    def __init__(self, save_dir='checkpoints', freq=20, verbose=False,
                 save_generator=True, save_discriminator=True, epoch_fmt = '05d'):
        """
        Periodically saves the generator and discriminator as .h5 files.

        Args:
            save_dir (str): Directory in which to save the models. Will create
                the directory if it does not already exist.
                Defaults to 'gan_checkpoints'.
            save_freq (int): How often to save the model (in units of epochs).
                Defaults to saving every 20 epochs.
            verbose (str): Whether to include a print statement as each model is saved.
                Default is False.
            save_generator (bool): Whether to save the generator. Default is True.
            save_discriminator (bool): Whether to save the discriminator. Default is True.
            epoch_fmt (str): String format to use for the epoch. Defaults to '05d'.
        """
        self.save_dir = save_dir
        self.freq = freq
        self.verbose = verbose
        self.save_generator = save_generator
        self.save_discriminator = save_discriminator
        self.epoch_fmt = epoch_fmt

    def on_train_begin(self, logs=None):
        if self.save_generator or self.save_discriminator:
            try:
                os.mkdir(self.save_dir)
            except FileExistsError:
                pass

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq == 0:
            if self.save_generator:
                self.model.generator.save(os.path.join(
                    self.save_dir,
                    f'generator_{epoch:{self.epoch_fmt}}.h5')
                )
            if self.save_discriminator:
                self.model.discriminator.save(os.path.join(
                    self.save_dir,f'discriminator_{epoch:{self.epoch_fmt}}.h5'))
            if self.verbose and (self.save_discriminator or self.save_generator):
                print(f'Model saved (epoch {epoch})')


# ----------------------------------------------------------------------------
# MISC. HELPERS
# ----------------------------------------------------------------------------


def get_figure_dimensions(num_subplots):
    """
    Returns somewhat aesthetically pleasing figure dimensions that are
    suitable for arranging the given number of subplots. Does so by
    converging to the middle two integer factors of the given number,
    e.g 40 has factors 1, 2, 4, 5, 8, 10, 20, 40 so get_figure_dimensions(40)
    should return (5,8).

    Args:
        num_subplots (int): Number of subplots to find dimensions for.
            Note odd values are incremented by 1.
    
    Returns:
        Tuple of integers for the figure width and height.
    """
    if num_subplots % 2 != 0:
        num_subplots += 1

    i = 2
    j = num_subplots // i
    ibest, jbest = i, j

    while j > i:
        i += 1
        j = num_subplots // i
        if i*j != num_subplots:
            continue
        ibest, jbest = i, j

    return ibest,jbest