"""Carl Granström, Convolutional Neural Network for MNIST -
The code has some redundant exports and similar because it was used to experiment with several different activations,
initializers, pooling layers and dropout values, some parameters might not be optimal, but the network currently
 has a best performance of 0.27% errors on the validation set, which is quite sufficient"""

import numpy
import h5py
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import AlphaDropout
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.initializers import lecun_uniform
from keras import backend as K
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

'''# fixed random seed for reproducibility - activate while testing
seed = 7
numpy.random.seed(seed)'''

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
#?
# model definition
def cnn_model():
	# create model
	model = Sequential()
	BatchNormalization()
	model.add(ZeroPadding2D(padding=(2, 2), data_format=None, input_shape=(1, 28, 28)))
	model.add(Conv2D(32, (5, 5), activation='relu', use_bias=False))
	BatchNormalization(axis=-1)
	model.add(Conv2D(32, (5, 5), activation='relu', use_bias=False))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))

	model.add(Conv2D(64, (3, 3), activation='relu', use_bias=False))
	BatchNormalization(axis=-1)
	model.add(Conv2D(64, (3, 3), activation='relu', use_bias=False))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.3))

	model.add(Flatten())
	BatchNormalization()

	model.add(Dense(512, activation='selu',kernel_initializer=lecun_uniform(seed=None), use_bias=True,
					bias_initializer=lecun_uniform(seed=None)))
	BatchNormalization()
	model.add(AlphaDropout(0.50))

	model.add(Dense(100, activation='selu',kernel_initializer=lecun_uniform(seed=None), use_bias=True,
					bias_initializer=lecun_uniform(seed=None)))
	model.add(AlphaDropout(0.50))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.summary()

	return model

# build the model
model = cnn_model()

# serialize model to JSON
model_json = model.to_json()
with open("SaveModel/model.json", "w") as json_file:
    json_file.write(model_json)

# Generator for randomization of image input
gen = ImageDataGenerator(rotation_range=12, width_shift_range=0.08, shear_range=0.1,
						 height_shift_range=0.08, zoom_range=0.08, horizontal_flip=True)

test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, y_train, batch_size=512)
test_generator = test_gen.flow(X_test, y_test, batch_size=512)
# model.fit(X_train, Y_train, batch_size=128, nb_epoch=1, validation_data=(X_test, Y_test))

# if the validation loss decreased after a epoch: save the model
checkpointer = ModelCheckpoint(filepath='SaveWeights/MNISTCNNweights.hdf5', monitor='val_acc', mode='auto', verbose=1,
							   save_best_only=True)

''' Does not work well for MNIST because of standardization of numbers
history = model.fit_generator(train_generator, steps_per_epoch=60000 // 512, epochs=150,
					validation_data=test_generator, validation_steps=10000 // 512, callbacks=[checkpointer])'''



# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=2500, verbose=2,
		  callbacks=[checkpointer])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
