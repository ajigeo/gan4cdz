import numpy as np
from numpy import load
from numpy import ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
from keras.layers.merge import add
#%%
def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = 2*((X1-np.min(X1))/(np.max(X1)-np.min(X1)))-1
    X2 = 2*((X2-np.min(X2))/(np.max(X2)-np.min(X2)))-1
    return [X1, X2]
#%%
def plot_images(data,num):
    pyplot.suptitle('NDWI VV image pair of index ' + str(num))
    pyplot.subplot(1,2,1)
    pyplot.imshow(data[0][num])
    pyplot.title('VV')

    pyplot.subplot(1,2,2)
    pyplot.imshow(data[1][num])
    pyplot.title('NDWI')
    pyplot.show() 
#%%
def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model
#%% 
# ResNet
# define an encoder block
def encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add downsampling layer
    g = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def resnet_decoder(layer_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # add upsampling layer
    g = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    # add batch normalization
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# function for creating an identity residual module
def residual_module(layer_in, n_filters):
    # conv1,
    init = RandomNormal(stddev=0.02)
    conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer=init)(layer_in)
    conv1 = BatchNormalization()(conv1, training=True)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    # conv2
    conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer=init)(conv1)
    conv2 = BatchNormalization()(conv2, training=True)
    # add filters, assumes filters/channels last
    layer_out = add([conv2, layer_in])
    # activation function
    layer_out = LeakyReLU(alpha=0.2)(layer_out)
    return layer_out

def resnet_generator(image_shape=(128,128,1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = encoder_block(in_image, 64)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 128)
    e4 = encoder_block(e3, 256)
    e5 = encoder_block(e4, 256)
    
    r1 = residual_module(e5,256)
    r2 = residual_module(r1,256)
    r3 = residual_module(r2,256)
    r4 = residual_module(r3,256)
    r5 = residual_module(r4,256)
    r6 = residual_module(r5,256)
    r7 = residual_module(r6,256)
    r8 = residual_module(r7,256)
    r9 = residual_module(r8,256)
    
    d1 = resnet_decoder(r9,256)
    d2 = resnet_decoder(d1,256)
    d3 = resnet_decoder(d2, 128)
    d4 = resnet_decoder(d3, 128)
    d5 = resnet_decoder(d4, 64)
    g = Conv2DTranspose(1, (4, 4), strides=(1, 1), padding='same', kernel_initializer=init)(d5)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model
#%%
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model
#%%
# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    y = y * 0.9
    return [X1, X2], y
#%%
# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = ones((len(X), patch_shape, patch_shape, 1))
    y = y * 0.1
    return X, y
#%%
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    #X_realA = (X_realA + 1) / 2.0
    #X_realB = (X_realB + 1) / 2.0
    #X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%06d.h5' % (step + 1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
#%%
# train pix2pix models
def train(d_model, g_model, gan_model, dataset, n_epochs=400, n_batch=1):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset)
#%%
dataset = load_real_samples('E:/vv8_ndwi.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator((128,128,1))
g_model = resnet_generator((128,128,1))
# define the composite model
gan_model = define_gan(g_model, d_model, (128,128,1))
train(d_model, g_model, gan_model, dataset)
#%%
plot_model(d_model, to_file='multiple_inputs.png', show_shapes=True,show_layer_names=False)
plot_model(g_model, to_file='resnet-new.png', show_shapes=True,show_layer_names=False)
#%%
[X1, X2] = load_real_samples('E:/vv8_ndwi.npz')
#%%
# load model
import tensorflow as tf
model = tf.keras.models.load_model('C:/Users/admin/resnet-400/model_533600.h5')
#%%
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
#%%
# generate image from source
gen_image = model.predict(src_image)
gen_image = gen_image.reshape(128,128)
src_image = src_image.reshape(128,128)
tar_image = tar_image.reshape(128,128)
#%% 
import matplotlib.pyplot as plt
def plot_final(x,y,z):
    plt.subplot(1,3,1)
    plt.imshow(x)
    plt.title('Source VV image')
    
    plt.subplot(1,3,2)
    plt.imshow(y)
    plt.title('Generated NDWI image')
    
    plt.subplot(1,3,3)
    plt.imshow(z)
    plt.title('Target NDWI image')

    plt.show()
#%%
# plot all three images
plot_final(src_image, gen_image, tar_image)
