# KerasGAN

***
Training a GAN is as easy as one, two, three!

```python
from kerasgan import GAN, GANSnapshot, GANCheckpoint

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
### Description
This is a simple all-in-one utility that provides `keras.Model` classes for training
regular generative adversarial networks (GANs), as well as conditional GANs (CGANs).
It is intended to be generic and easy to use.

`kerasgan` is essentially an engine. You'll need to supply models for the generator and discriminator 
(and of course provide some training data!), but the training itself is handled by the GAN classes.

<p>&nbsp;</p>

### Requirements
- TensorFlow (tested on version 2.7 and above)
- Jupyter / IPython (recommended)

<p>&nbsp;</p>

### Getting Started
Check out the two example notebooks for quickly training an example GAN and CGAN with MNIST.

For tips on crafting generators and discriminators, check out these resources:
- [TensorFlow DCGAN tutorial](https://www.tensorflow.org/tutorials/generative/dcgan)
- [keras.io GAN example](https://keras.io/examples/generative/dcgan_overriding_train_step/)

<p>&nbsp;</p>

### Callbacks
Note: The display callbacks are intended to be used in an interactive environment (Jupyter, Colab, etc).

`kerasgan` also includes several callbacks for real-time visualisation and model checkpointing.
There are separate *snapshot* callbacks for GANs and CGANs as the CGANs require labels. If you're
training a GAN, use `GANSnapshot`. If you're training a CGAN, use `CGANSnapshot` or `CGANClassSnapshot`.
Mixing these up could have terrible consequences.


#### List of Callbacks


`GANSnapshot` creates example figures after each epoch to visualise the generator's output.
You can optionally provide a seed, otherwise a random seed is created at the start of training.

![output_snapshot](https://user-images.githubusercontent.com/13238315/186830859-aad174a5-01c0-4f7a-af5a-5a2f05da04c5.png)

`CGANSnapshot` is the same as `GANSnapshot` but also incorporates labels.

<p>&nbsp;</p>

`CGANClassSnapshot` will create rows of examples, with each column displaying a different class (i.e. label).

![output_class_snapshot](https://user-images.githubusercontent.com/13238315/186831013-947228d6-670b-41ce-bc58-5ff9a33f104b.png)

You can reverse these roles by passing `columns_indicate = 'examples'`, i.e. each row now shows a different class and each column shows a different example.

![output_class_snapshot_rowclass](https://user-images.githubusercontent.com/13238315/186831047-c2ac9206-fe7c-4a04-89ac-2d180997e376.png)

<p>&nbsp;</p>

Lastly, `GANCheckpoint` will periodically save the generator and/or discriminator.
