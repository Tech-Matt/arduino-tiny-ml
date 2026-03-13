#include <Arduino_LSM9DS1.h>
#include <math.h>
#include <arduinoFFT.h>


const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 128;
int samplesRead = numSamples;
// the following tracks which gesture iteration we are in
int sampleCount = 1;

// Buffer to hold 1 window of data for 6 axes (aX, aY, aZ, gX, gY, gZ)
float windowData[6][numSamples];

// Feature Array to hold the 6 calculated features for each of the 6 axes (36 total)
float features[6][6];

String axisNames[6] = {"aX", "aY", "aZ", "gX", "gY", "gZ"};
String featureNames[6] = {"Mean", "Std", "RMS", "Min", "Max", "Power"};

arduinoFFT FFT = arduinoFFT();

void setup() {
  Serial.begin(115200);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Print the header at startup
  Serial.println("sample, feature, aX, aY, aZ, gX, gY, gZ");
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Temporary
  samplesRead = 0;

  // wait for significant motion
  // while (samplesRead == numSamples) {
  //   if (IMU.accelerationAvailable()) {
  //     // read the acceleration data
  //     IMU.readAcceleration(aX, aY, aZ);

  //     // sum up the absolutes to detect movement regardless of direction
  //     float aSum = fabs(aX) + fabs(aY) + fabs(aZ);

  //     // check if it's above the threshold
  //     if (aSum >= accelerationThreshold) {
  //       // reset the sample read count
  //       samplesRead = 0;
  //       break;
  //     }
  //   }
  // }

  // DATA COLLECTION
  // check if the all the required samples have been read since
  // the last time the significant motion was detected
  while (samplesRead < numSamples) {
    // check if both new acceleration and gyroscope data is
    // available
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      // read the acceleration and gyroscope data
      IMU.readAcceleration(windowData[0][samplesRead], windowData[1][samplesRead], windowData[2][samplesRead]);
      IMU.readGyroscope(windowData[3][samplesRead], windowData[4][samplesRead], windowData[5][samplesRead]);
      samplesRead++;

    }
  }

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

    // Calculate sums, min, max
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

    // calculate variance for standard deviation
    float varianceSum = 0;
    for (int i = 0; i < numSamples; i++) {
      varianceSum += pow(windowData[axis][i] - mean, 2);
    }
    float stdDev = sqrt(varianceSum / numSamples);

    // PSD Calculation
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

  // Array of feature names
  const char* featureNames[6] = {"Mean", "Std", "RMS", "Min", "Max", "PSD"};
  // Print as CSV format
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

  sampleCount++;
  // Delay before next sample
  delay(2000);
}
