# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:06:47 2019

@author: Asier
"""

import numpy as np
import pandas as pd

# Keras imports
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import concatenate
from keras.optimizers import adam

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def build_model_cnn_2(input_shape):
    model = Sequential()

    model.add(Conv2D(256, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    optimizer = adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def build_model_cnn(input_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(9, activation='softmax'))

    optimizer = adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def build_ensemble_model(models):
    for i, model in enumerate(models):
        for layer in model.layers:
            layer.trainable = False
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
    ensemble_visible = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(128, activation='sigmoid')(merge)
    output = Dense(9, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    # Parameters
    batch_size = 32
    epochs = 50
    k = 5
    n_models = 3

    x = np.load('../input/X_train_kaggle.npy')
    y_df = pd.read_csv('../input/y_train_final_kaggle.csv')
    y_labels = y_df['Surface'].values

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(y_labels)

    scalers = {}
    for i in range(x.shape[1]):
        scalers[i] = StandardScaler()
        x[:, i, :] = scalers[i].fit_transform(x[:, i, :])

    x = np.expand_dims(x, axis=-1)
    input_shape = x.shape[1::]

    skf = StratifiedKFold(n_splits=k)

    cvscores = np.zeros((k, n_models))
    train_cvscores = np.zeros((k, n_models))

    k_index = 0
    for train_index, test_index in skf.split(x, y_labels):
        models = [build_model_cnn(input_shape), build_model_cnn_2(input_shape)]

        if len(models) != n_models - 1:
            raise Exception('Number of models does not coincide with n_models')

        x_train_model = x[train_index]
        y_train_model = y_labels[train_index]

        #x_train_model, x_train_ensemble, y_train_model, y_train_ensemble = train_test_split(x[train_index],
        #                                                                                    y_labels[train_index],
        #                                                                                    test_size=0.3,
        #                                                                                    random_state=42)
        y_train_model = label_binarizer.transform(y_train_model)
        #y_train_ensemble = label_binarizer.transform(y_train_ensemble)

        y_test = label_binarizer.transform(y_labels[test_index])

        for model_index, model in enumerate(models):
            train_scores = model.fit(x_train_model, y_train_model, batch_size=batch_size,
                                     epochs=epochs, verbose=0)
            scores = model.evaluate(x[test_index], y_test, verbose=0)
            print("Model " + str(model_index) + ": Train -> %s: %.2f%%" % (model.metrics_names[1],
                                                                           train_scores.history['acc'][-1] * 100))
            print("Model " + str(model_index) + ": Test  -> %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            print("-------------------")
            cvscores[k_index, model_index] = scores[1] * 100
            train_cvscores[k_index, model_index] = train_scores.history['acc'][-1] * 100

        ensemble_model = build_ensemble_model(models)

        train_scores = ensemble_model.fit([x_train_model for _ in models], y_train_model, batch_size=batch_size,
                                          epochs=epochs, verbose=0)
        scores = ensemble_model.evaluate([x[test_index] for _ in models], y_test, verbose=0)
        print("Ensemble Model " + str(n_models) + ": Train -> %s: %.2f%%" % (ensemble_model.metrics_names[1],
                                                                             train_scores.history['acc'][-1]
                                                                             * 100))
        print("Ensemble Model " + str(n_models) + ": Test  -> %s: %.2f%%" % (ensemble_model.metrics_names[1],
                                                                             scores[1] * 100))
        print("-------------------")

        cvscores[k_index, n_models - 1] = scores[1] * 100
        train_cvscores[k_index, n_models - 1] = train_scores.history['acc'][-1] * 100

        k_index += 1

    for model_index in range(n_models):
        print("Model " + str(model_index) + ": Train -> %.2f%% (+/- %.2f%%)" % (np.mean(train_cvscores[:, model_index]),
                                                                                np.std(train_cvscores[:, model_index])))
        print("Model " + str(model_index) + ": Test  -> %.2f%% (+/- %.2f%%)" % (np.mean(cvscores[:, model_index]),
                                                                                np.std(cvscores[:, model_index])))

    x = np.load('../input/X_test_kaggle.npy')

    for i in range(x.shape[1]):
        x[:, i, :] = scalers[i].transform(x[:, i, :])

    x = np.expand_dims(x, axis=-1)

    y_pred = ensemble_model.predict([x for _ in range(2)])
    y_pred = label_binarizer.inverse_transform(y_pred)

    out_df = pd.DataFrame(columns=['# Id', 'Surface'])
    out_df['# Id'] = range(1705)
    out_df['Surface'] = y_pred
    out_df.to_csv('prediction.csv', index=False)