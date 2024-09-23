# Ball-screw-fault-detection
The vibration data was captured using an NI-DAQ through a LabVIEW program. A deep learning model, developed using CNN, detects the machine's state as either 'good' or 'bad' based on the vibration data.

Methodology for the Above Project:

1) Vibration data from the feed drive setup is acquired using an NI-DAQ with an IEPE sensor.
2) Initially, the model was developed using training data, which falls into two categories: 'Good' and 'Bad.' The 'Good' data was acquired when the system was in ideal running condition, while the 'Bad' data was collected when the ball screw was dented. The vibration data was plotted in STFT format, and the corresponding STFT images were saved. These images were then used to develop the CNN classification model.
3) In real-time, the acquired data will be plotted as STFT and fed into the model for prediction. The model will predict the condition of the ball screw as either good or bad.
4) The system acquires data and checks the system condition every 2 minutes.
