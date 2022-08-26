# kerasgan
Custom Keras models for training GANs and CGANs.
***
Training a GAN is as easy as one, two, three!
```python
# 1. Instantiate
gan = GAN(
    generator=generator,
    discriminator=discriminator,
    latent_dim=LATENT_DIM
)

# 2. Compile
gan.compile(
    generator_optimizer=keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.05),
    discriminator_optimizer=keras.optimizers.Adam(learning_rate=2e-4,beta_1=0.05)
)

# 3. Fit!
hist = gan.fit(
    image_data, epochs=50, verbose=2, callbacks = [GANSnapshot(), GANCheckpoint()]
)
```

### Requirements
- Tensorflow (tested on version 2.7 and above)
- Jupyter / Ipython (recommended)

### Description
This is a simple all-in-one utility that provides keras.Model classes for training
regular generative adversarial networks (GANs), as well as conditional GANs (CGANs).
It is intended to be generic and easy to use. `kerasgan` is essentially an engine:
you'll need to create a generator, discriminator, and of course provide some training data!

### Getting Started
Check out the two example notebooks for quickly training an example GAN and CGAN with MNIST.

For tips on crafting generators and discriminators, check out these resources:
- [Tensorflow DCGAN tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- [keras.io GAN example](https://keras.io/examples/generative/dcgan_overriding_train_step/)

### Callbacks
Note: The callbacks are intended to be used in an interactive environment (Jupyter, Colab, etc)

`kerasgan` also includes several callbacks for real-time visualisation and model checkpointing.
There are separate *snapshot* callbacks for GANs and CGANs as the CGANs require labels. If you're
training a GAN, use `GANSnapshot`. If you're training a CGAN, use `CGANSnapshot` or `CGANClassSnapshot`.
Mixing these up could have terrible consequences.

These callbacks are:

`GANSnapshot`: Creates example figures after each epoch to visualise the generator's output.
You can optionally provide a seed, otherwise a random seed is created at the start of training.

`CGANSnapshot`: Same as `GANSnapshot` but utilises labels.

`CGANClassSnapshot`: This callback will create rows of examples, with each column displaying a different class (i.e. label).
You can reverse these roles by passing `columns_indicate = 'examples'`.

`GANCheckpoint`: Periodically saves the generator and/or discriminator.
