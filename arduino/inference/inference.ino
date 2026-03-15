/*
  IMU Classifier
  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.
  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.
  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.
  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry
  This example code is in the public domain.
*/


#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
//#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
//#include <tensorflow/lite/version.h>
#include <arduinoFFT.h>
#include "model.h"

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 128;
int samplesRead = numSamples;

// Buffer to hold 1 window of data for 6 axes (aX, aY, aZ, gX, gY, gZ)
float windowData[6][numSamples];

// Feature Array to hold the 6 calculated features for each of the 6 axes (36 total)
float features[6][6];

const float FEATURE_MIN[6][6] = {
  // Mean, Std, RMS, Min, Max, PSD
  {-4.0, 0.0, 0.0, -4.0, -4.0, 0.0},
  {-4.0, 0.0, 0.0, -4.0, -4.0, 0.0},
  {-4.0, 0.0, 0.0, -4.0, -4.0, 0.0},
  {-2000.0, 0.0, 0.0, -2000.0, -2000.0, 0.0},
  {-2000.0, 0.0, 0.0, -2000.0, -2000.0, 0.0},
  {-2000.0, 0.0, 0.0, -2000.0, -2000.0, 0.0}
};

const float FEATURE_MAX[6][6] = {
  // Mean, Std, RMS, Min, Max, PSD
  {4.0, 4.0, 4.0, 4.0, 4.0, 128000000.0},
  {4.0, 4.0, 4.0, 4.0, 4.0, 128000000.0},
  {4.0, 4.0, 4.0, 4.0, 4.0, 128000000.0},
  {2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 128000000.0},
  {2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 128000000.0},
  {2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 128000000.0}
};

// Initialize Fast Fourier Transform Library
arduinoFFT FFT = arduinoFFT();

// global variables used for TensorFlow Lite (Micro)
//tflite::MicroErrorReporter tflErrorReporter;

// TFLM = TensorFlowLiteforMicrocontrollers
// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
// What an operation resolver does, is filling a dictionary with every single math
// function Tensorflow knows about, it then links it to the model.h which is "operation
// implementation agnostic"
tflite::AllOpsResolver tflOpsResolver;

// This initializes the model that will be read from the model.h file
const tflite::Model* tflModel = nullptr;

// Initializes the Interpreter. This is the core engine that actually
// runs the model. It manages the memory and performs the operation on tflModel using
// operations from tflOpsResolver
tflite::MicroInterpreter* tflInterpreter = nullptr;

// A tensor is simply a multidimensional array. This will act as the gateway to where
// we feed the IMU sensor data (after preprocessing / normalization)
TfLiteTensor* tflInputTensor = nullptr;
// This is where the model is going to put the output tensor
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
// The Interpreter uses this arena (just a memory block in RAM) to store
// input data, output data and intermediate math calculations.
// the __attribute__ tells the Arduino to make sure that the starting memory address
// of the arena is an exact multiple of 16. Why? TFLM is highly optimized for
// microcontrollers. Under the hoods it uses advanced processor instructions like SIMD.
// Those operations usually strictly requires the data they are working with to be
// structurally aligned in memory in 16 byte chunks.
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "Rest",
  "Left-right",
  "Down-Up",
  "Circle"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1); // This safely halts the progam
  }


  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  // Following check is to prevent version mismatch between Tensorflow versions in
  // Colab and TFLite, in the way they pack data in the model.h file
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1); // This safely halts the program
  }

  // Build an interpreter to run the model with.
  // This links together all the other TFL components that we have
  // Instantiated before
  static tflite::MicroInterpreter static_interpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter = &static_interpreter;

  // Create an interpreter to run the model (Alternative way, but dangerous)
  // This could cause HEAP FRAGMENTATION
  //tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1); // This safely halts the program
  }

  // This connects the input and output tensors global variables to
  // the Interpreter. The '0' here just means first input and output, this is 
  // because different neural networks could have more input/output tensors. 
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // COLLECT SAMPLES
  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
    // check if both new acceleration and gyroscope data is available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(windowData[0][samplesRead], windowData[1][samplesRead], windowData[2][samplesRead]);
      IMU.readGyroscope(windowData[3][samplesRead], windowData[4][samplesRead], windowData[5][samplesRead]);
      samplesRead++;
    }
  }

  // Reset samples read
  samplesRead = 0;
  
  // FEATURE EXTRACTION
  // Now that the buffer is full calculate the math for each of the 6 axes
  for (int axis = 0; axis < 6; axis++) {
    float sum = 0;
    float sumSquares = 0;
    float minVal = windowData[axis][0];
    float maxVal = windowData[axis][0];

    // arrays required by FFT
    double vReal[numSamples];
    double vImag[numSamples];

    // SUMS, MIN, MAX
    for (int i = 0; i < numSamples; i++) {
      float val = windowData[axis][i];
      sum += val;
      sumSquares += (val * val);
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;

      vReal[i] = (double)val;
      vImag[i] = 0.0;
    }

    float mean = sum / numSamples;
    float rms = sqrt(sumSquares / numSamples);

    // VARIANCE
    float varianceSum = 0;
    for (int i = 0; i < numSamples; i++) {
      varianceSum += pow(windowData[axis][i] - mean, 2);
    }
    float stdDev = sqrt(varianceSum / numSamples);

    // PSD 
    // Apply a Hamming Window to smooth edges of data block
    FFT.Windowing(vReal, numSamples, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    // Compute FFT
    FFT.Compute(vReal, vImag, numSamples, FFT_FORWARD);
    // Convert complex numbers to magnitudes
    FFT.ComplexToMagnitude(vReal, vImag, numSamples);

    // Find peak PSD
    double peakPSD = 0;
    for (int i = 1; i < (numSamples / 2); i++) {
      // PSD is the magnitude squared
      double power = vReal[i] * vReal[i];
      if (power > peakPSD) {
        peakPSD = power;
      }
    }


    // Store calculated features into the matrix
    features[axis][0] = mean;
    features[axis][1] = stdDev;
    features[axis][2] = rms;
    features[axis][3] = minVal;
    features[axis][4] = maxVal;
    features[axis][5] = peakPSD;
  }

    // // Normalize the peak PSD
    // peakPSD = peakPSD / numSamples;
  // Array of feature names
  const char* featureNames[6] = {"Mean", "Std", "RMS", "Min", "Max", "PSD"};
  // Print as CSV format
  /*
  for (int feat = 0; feat < 6; feat++) {
    // print sample number first
    Serial.print(sampleCount);
    Serial.print(", ");

    // print feature name 
    Serial.print(featureNames[feat]);
    Serial.print(", ");
    // Print 6 axis values for this feature
    for (int axis = 0; axis < 6; axis++) {
      // Print the value with 4 decimal places
      Serial.print(features[axis][feat], 4);
      // Print a comma unless it's the last column
      if (axis < 5) {
        Serial.print(",");
      }
    }
    Serial.println(); // Move to the next feature row
  }
  */

  // NORMALIZE THE FEATURES, THEN PROVIDE INPUT TENSOR TO THE MODEL
  int tensorIndex = 0;
  for (int feat = 0; feat < 6; feat++) {
    for (int axis = 0; axis < 6; axis++) {
      float rawValue = features[axis][feat];
      float range = FEATURE_MAX[axis][feat] - FEATURE_MIN[axis][feat];
      float normalizedValue = (rawValue - FEATURE_MIN[axis][feat]) / range;

      // Ensure the normalized value is clamped between 0 and 1
      if (normalizedValue < 0.0) normalizedValue = 0.0;
      if (normalizedValue > 1.0) normalizedValue = 1.0;

      // Store in input tensor
      tflInputTensor->data.f[tensorIndex] = normalizedValue;
      tensorIndex++;
    }
  }

  // Run inferencing
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    while (1);
    return;
  }

  // Loop through the output tensor values from the model
  int highest_index = 0;
  float highest_value = 0;
  for (int i = 0; i < NUM_GESTURES; i++) {
    Serial.print(GESTURES[i]);
    Serial.print(": ");
    float predicted = tflOutputTensor->data.f[i];
    Serial.println(predicted, 6);

    // Compute the predicted index value
    if (predicted > highest_value) {
      highest_value = predicted;
      highest_index = i;
    }
  }

  Serial.println("Prediction: ");
  Serial.println( GESTURES[highest_index]);
  Serial.println();
  delay(2000);
}