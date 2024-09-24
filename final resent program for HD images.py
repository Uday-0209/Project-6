import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.colors as colors
import scipy.signal as signal
from matplotlib import pyplot as plt
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input

def stft_predict():
    data = tf.keras.utils.image_dataset_from_directory('D:\\Model\\data', batch_size=8)

    data_iterator = data.as_numpy_iterator()

    batch = data_iterator.next()


    data = data.map(lambda x, y: (tf.image.resize(x, (500, 900)), y))

    print(data)


    data.as_numpy_iterator().next()

    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    test_size = int(len(data) * 0.1) + 1

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    resnet_model = Sequential()

    pretrained_model = ResNet101(include_top=False, input_shape=(500, 900, 3), pooling='avg', classes=2, weights='imagenet')
    for layer in pretrained_model.layers:
        layer.trainable = False
    resnet_model.add(pretrained_model)
    resnet_model.add(Flatten())
    resnet_model.add(Dense(256, activation='relu'))
    resnet_model.add(Dense(64 , activation = 'relu'))
    resnet_model.add(Dense(4, activation='relu'))  # Set the number of classes to 4
    resnet_model.add(Dense(2, activation = 'softmax'))

    resnet_model.compile('Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    logdir = 'D:\\Model\\logs'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model = resnet_model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator():
        X, y = batch
        y_onehot = tf.one_hot(y , depth = 2)
        yhat = resnet_model.predict(X)
        print(yhat)
        pre.update_state(y_onehot, yhat)
        re.update_state(y_onehot, yhat)
        acc.update_state(y_onehot, yhat)

    print('The precision value:', pre.result().numpy())
    print('The recall value:', re.result().numpy())
    print('The accuracy value:', acc.result().numpy())

    with open('D:\Model\model_file.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    def calcSTFT_norm(df, column_name, samplingFreq=10000, window='hann', nperseg=50000, figsize=(9, 5), cmap='plasma',
                      ylim_max=None, save_path=None):
        inputSignal = df[column_name].values

        if len(inputSignal) < nperseg:
            print(
                f"Warning: nperseg = {nperseg} is greater than input length = {len(inputSignal)}. Skipping STFT calculation.")
            return

        f, t, Zxx = signal.stft(inputSignal, samplingFreq, window=window, nperseg=nperseg)

        fig = plt.figure(figsize=figsize)
        spec = plt.pcolormesh(t, f, np.abs(Zxx),
                              norm=colors.PowerNorm(gamma=1. / 6.),
                              cmap=plt.get_cmap(cmap))
        cbar = plt.colorbar(spec)

        ax = fig.axes[0]
        ax.set_xlim(0, t[-1])

        plt.title(f'STFT Spectrogram for {column_name}')
        ax.grid(True)
        ax.set_title('STFT Magnitude')
        if ylim_max:
            ax.set_ylim(0, ylim_max)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')

        if save_path:
            plt.savefig(save_path)
            plt.show()

    file_name = 'D:\\Model\\GD_1000_60_12_1.csv'
    df = pd.read_csv(file_name)

    column_names = ['cDAQ9189-20796A8Mod1_ai0', 'cDAQ9189-20796A8Mod1_ai1', 'cDAQ9189-20796A8Mod1_ai2',
                    'cDAQ9189-20796A8Mod1_ai3']

    for i, column_name in enumerate(column_names, start=1):
        base_save_path = f'output{i}'
        save_path = f'D:\model\{base_save_path}_stft.png'
        calcSTFT_norm(df, column_name, save_path=save_path)

    image_paths = [
        'D:\\Model\\output1_stft.png',
        'D:\\Model\\output2_stft.png',
        'D:\\Model\\output3_stft.png',
        'D:\\Model\\output4_stft.png'
    ]

    with open('D:\\Model\\model_file.pkl', 'rb') as modelfile:
        new_model = pickle.load(modelfile)



    results = []  # Store the results for each image

    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(500, 900))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        yhat = resnet_model.predict(img_array)
        print(yhat)
        yhatt = float(yhat[0][0])
        print(yhatt)

        if yhatt > 0.5:
            predicted_class = 'good'

        else:
            predicted_class = 'bad'

        results.append(predicted_class)

    return results


# Call the function to get the results
results = stft_predict()

for predicted_class in results:
    print(f'The data  is {predicted_class}')



