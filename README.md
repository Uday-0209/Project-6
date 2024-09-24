# Ball-screw-fault-detection
The vibration data was captured using an NI-DAQ through a LabVIEW program. A deep learning model, developed using CNN, detects the machine's state as either 'good' or 'bad' based on the vibration data.

Methodology for the Above Project:

1) Vibration data from the feed drive setup is acquired using an NI-DAQ with an IEPE sensor.
2) Initially, the model was developed using training data, which falls into two categories: 'Good' and 'Bad.' The 'Good' data was acquired when the system was in ideal running condition, while the 'Bad' data was collected when the ball screw was dented. The vibration data was plotted in STFT format, and the corresponding STFT images were saved. These images were then used to develop the CNN classification model.
3) In real-time, the acquired data will be plotted as STFT and fed into the model for prediction. The model will predict the condition of the ball screw as either good or bad.
4) The system acquires data and checks the system condition every 2 minutes.

Here we developed the model for both the testrig and Spindle set-up.The below images are the frontend and backend of the labview program.
![Screenshot 2024-09-24 093317](https://github.com/user-attachments/assets/ae43c21c-5740-47fc-9c13-9623bad74f71)

This is frontend for Testrig condition monitoring.

![test rig ml model backend](https://github.com/user-attachments/assets/d62afe15-445e-4074-bf89-79ba15e98912)

The above image is the backend of the Testrig condition monitoring.

![spindle motor prediction](https://github.com/user-attachments/assets/ed82adf2-456b-4db1-a9e4-0c497054d500)

This is frontend of Spindle condition monitoring.

![spindle motor backend](https://github.com/user-attachments/assets/7c5c5f62-f09b-4d85-ae35-1baf2e70b091)

The backend of the Spindle condition monitoring.
