import cv2
import os
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
from tensorflow.keras.models import load_model
import time

start_time = time.time()
def stft_predict():
    # normalized_path = os.path.normpath(input_path)
    # print(normalized_path)
    # # Replace single backslashes with double backslashes
    # file_name = normalized_path.replace(os.path.sep, os.path.sep + os.path.sep)
    # print(file_name)

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
            #plt.show()

    file_name = 'D:\\Model\\vibration data\\waveform data.csv'
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

    # with open('D:\\Model\\model_file.pkl', 'rb') as modelfile:
    #     new_model = pickle.load(modelfile)

    new_model = load_model('D:\\Model\\working_model.h5')


    results = []  # Store the results for each image

    for image_path in image_paths:
        img = image.load_img(image_path, target_size=(500, 900))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        yhat = new_model.predict(img_array)
        print(yhat)
        yhatt = float(yhat[0][0])
        yhat2 = float(yhat[0][1])
        print(yhat2)
        print(yhatt)


        if yhatt > yhat2:
            predicted_class = 'bad'

        else:
            predicted_class = 'good'

        results.append(predicted_class)

    return results


# Call the function to get the results
results = stft_predict()

for predicted_class in results:
    print(f'The data  is {predicted_class}')

end_time = time.time()
running_time = end_time - start_time

print(f"Program running time: {running_time} seconds")
