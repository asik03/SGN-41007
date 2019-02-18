# -*- coding: utf-8 -*-
import traffic_signs
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.applications import vgg16


num_classes = 2
batch_size = 32
epochs = 20

X, y = traffic_signs.load_data('./input')

# Normalization
X_norm = ((X - np.amin(X)) / (np.amax(X)- np.amin(X)))

X_train, X_test, y_train, y_test = train_test_split(X_norm, y)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Model structure.
model = Sequential()
model.add(vgg16.VGG16(include_top=False, input_shape = (64, 64, 3), weights='imagenet', input_tensor=None, pooling=None, classes=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))
model.summary()


model.compile(loss=categorical_crossentropy,
              optimizer=SGD(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])




