# Постройте и обучите нейронную сеть на наборе данных с предметами одежды. Для этого создайте в коде три функции:
# загрузки обучающей выборки load_train(),
# создания модели create_model(),
# запуска модели train_model().
# Добейтесь того, чтобы значение accuracy на тестовой выборке было не меньше 85%.
# Отправьте ваш код в файле с расширением .py .

from keras.datasets import fashion_mnist
from keras.layers import Dense
from keras.models import Sequential
import numpy as np

def load_train(path):
    features_train = np.load(path + 'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    features_train = features_train.reshape(features_train.shape[0], 28 * 28) / 255.
    return features_train, target_train


def create_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=input_shape, activation='relu'))
    model.add(Dense(10, input_shape=input_shape, activation='softmax'))
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model


def train_model(model, train_data, test_data, batch_size=32, epochs=5,
               steps_per_epoch=None, validation_steps=None):

    features_train, target_train = train_data
    features_test, target_test = test_data
    model.fit(features_train, target_train, 
              validation_data=(features_test, target_test),
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2, shuffle=True)

    return model
