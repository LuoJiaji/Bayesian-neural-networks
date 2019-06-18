import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import distributions
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Input, Flatten, Dense, Dropout, Layer
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras import backend as K
from keras import activations, initializers
from keras import callbacks, optimizers

epochs = 30
train_size = 32
noise = 1.0

def mixture_prior_params(sigma_1, sigma_2, pi, return_sigma=False):
    params = K.variable([sigma_1, sigma_2, pi], name='mixture_prior_params')
    sigma = np.sqrt(pi * sigma_1 ** 2 + (1 - pi) * sigma_2 ** 2)
    return params, sigma

def log_mixture_prior_prob(w):
    comp_1_dist = distributions.Normal(0.0, prior_params[0])
    comp_2_dist = distributions.Normal(0.0, prior_params[1])
    comp_1_weight = prior_params[2]    
    return K.log(comp_1_weight * comp_1_dist.prob(w) + (1 - comp_1_weight) * comp_2_dist.prob(w))    

def neg_log_likelihood(y_true, y_pred, sigma=noise):
    # print(y_true.shape, y_pred.shape)
    # quit()
    dist = distributions.Normal(y_true, sigma)
    return K.sum(dist.log_prob(y_pred))

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)

# Mixture prior parameters shared across DenseVariational layer instances
prior_params, prior_sigma = mixture_prior_params(sigma_1=1.0, sigma_2=0.1, pi=0.2)

class DenseVariational(Layer):
    def __init__(self, output_dim, kl_loss_weight, activation=None, **kwargs):
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        self.activation = activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):  
        self._trainable_weights.append(prior_params) 

        self.kernel_mu = self.add_weight(name='kernel_mu', 
                                         shape=(input_shape[1], self.output_dim),
                                         initializer='uniform',
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', 
                                       shape=(self.output_dim,),
                                       initializer='uniform',
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                          shape=(input_shape[1], self.output_dim),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho', 
                                        shape=(self.output_dim,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, x):
        kernel_sigma = tf.nn.softplus(self.kernel_rho)
#        kernel = self.kernel_mu + 0.1*kernel_sigma * tf.random_normal(self.kernel_mu.get_shape())
        kernel = self.kernel_mu 

        bias_sigma = tf.nn.softplus(self.bias_rho)
#        bias = self.bias_mu + 0.1*bias_sigma * tf.random_normal(self.bias_mu.get_shape())
        bias = self.bias_mu
        
        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) + self.kl_loss(bias, self.bias_mu, bias_sigma))
        
        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def kl_loss(self, w, mu, sigma):
        variational_dist = distributions.Normal(mu, sigma)
        return kl_loss_weight * K.sum(variational_dist.log_prob(w) - log_mixture_prior_prob(w))


class MyLayer(Layer):

    def __init__(self, output_dim,activation=None, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.kernel_mu = self.add_weight(name='kernel_mu', 
                                         shape=(input_shape[1], self.output_dim),
                                         initializer='uniform',
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu', 
                                       shape=(self.output_dim,),
                                       initializer=initializers.constant(0.0),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho', 
                                          shape=(input_shape[1], self.output_dim),
                                          initializer='uniform',
#                                          initializer = initializers.normal(stddev=0.1),
                                          trainable=True)
        # self.bias_rho = self.add_weight(name='bias_rho', 
        #                                 shape=(self.output_dim,),
        #                                 initializer='uniform',
        #                                 trainable=True)
        
        super(MyLayer, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        kernel_sigma = tf.nn.softplus(self.kernel_rho)
        kernel = self.kernel_mu + 0.1 * kernel_sigma * tf.random_normal(self.kernel_mu.get_shape())
        
#         kernel = self.kernel_mu 
        
#        kernel = tf.random_normal([1], mean=self.kernel_mu, stddev=self.kernel_rho )
        
#        print(kernel)
        # bias_sigma = tf.nn.softplus(self.bias_rho)
        # bias = self.bias_mu + 0.1*bias_sigma * tf.random_uniform(self.bias_mu.get_shape())
        
        bias = self.bias_mu 
        
        return self.activation(K.dot(x, kernel) + bias)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    
batch_size = train_size
num_batches = train_size / batch_size
kl_loss_weight = 1.0 


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print('x train shape', x_train.shape)
print('y train shape', y_train.shape)
# quit()
input_shape = (28,28,1)

input_data = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
#x = Dense(128, activation='relu', name='fc2')(x)
# x = DenseVariational(10, kl_loss_weight=kl_loss_weight)(x)
#x = MyLayer(128,activation='relu')(x)
#x = MyLayer(128,activation='relu')(x)
x = MyLayer(10,activation='softmax')(x)
#x = Dense(10, activation='softmax', name='fc_10')(x)
model = Model(input_data, x)

# rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
# model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

# model.compile(loss = neg_log_likelihood, optimizer = optimizers.Adam(lr=0.03), metrics = ['accuracy'])
model.compile(loss = kullback_leibler_divergence, optimizer = optimizers.SGD(lr = 0.003), metrics = ['accuracy'])

#model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.Adam(lr=0.03), metrics = ['accuracy'])
#model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(), metrics = ['accuracy'])

model.summary()

# quit()

# history=model.fit(x_train, y_train , epochs=10)
model.fit(x_train, y_train, epochs=20, batch_size=128)
y_test = np.argmax(y_test, axis=1)

pre_cum = []
pre_arg = []
acc_cum = []

for i in range(10):
    print('\r','test iter:',i,end = '')
    pre = model.predict(x_test)
#    pre_arg += [np.argmax(pre, axis=1)]
    pre_cum += [pre]
    pre = np.argmax(pre, axis=1)
    pre_arg += [pre]
    acc = np.mean(pre==y_test)
    acc_cum += [acc]
print('\n')
#y_test = np.argmax(y_test, axis=1)
#acc = np.mean(pre==y_test)
#print('accuracy:',acc)
#model.save('./model/BNN.h5')

#noise = np.random.rand(28,28,1)
#plt.imshow(noise,cmap='gray')
#plt.show()


mean = np.random.rand(28,28)
std = np.random.rand(28,28)

img = np.random.rand(10,28,28,1)
pre_cum = []
pre_arg = []
acc_cum = []

for i in range(10):
    print('\r','test iter:',i,end = '')
    pre = model.predict(x_test)
#    pre_arg += [np.argmax(pre, axis=1)]
    pre_cum += [pre]
    pre = np.argmax(pre, axis=1)
    pre_arg += [pre]
    acc = np.mean(pre==y_test)
    acc_cum += [acc]
print('\n')