# arduino-tiny-ml
The repository contains a gesture recognition system implemented on Arduino nano 33 ble sense board using a neural network based on gyroscope and accelerometer inputs. It uses:
- Google Colab for training a TensorFlow Lite neural network and deploy the arduino header file containing the model.
- Arduino IDE for collecting the data in .csv files for the training and the inference.

# Training data collection

The arduino project [*input_capture*](arduino/input_capture/input_capture.ino) contains the source code that collects features from the gyroscope and accelerometer in widows of 128 samples:
- Mean value
- Standard deviation
- Root mean square
- Minimum value
- Maximum value
- Power Spectral Density 

These measurements are computed directly in the arduino microcontroller and saved in *.csv* files for seprate gestures, ready for the actual neural network training in google colab.
In our implementation, we considered four types of movements: 
- Rest (nothing)
- Shake (a left-right movement)
- Up-Down
- Circle 

The measurements are stored in the [training_data](training_data) filder. 

During training and inference, the Arduino board is handled with the usb port facing upwards and the chips of the board facing right. 
# Neural Network training 
The [colab notebook](GesturesRecognitionTraining.ipynb) contains the steps for the creation and validation of the model automatically using its confusion matrix. 

# Inference 
The arduino project [*training_data*](arduino/inference/inference.ino) contains the source code for the inference of the model. 
