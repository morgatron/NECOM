#ifndef _INC_CALIBRATION_PULSES
#define _INC_CALIBRATION_PULSES
#include "Arduino.h"
namespace calibration{
  const float SAMPLE_RATE =256;
  const int SIZE_CAL_WAVEFORM = 30000; // At least 256*80
  extern int16_t calibration_waveform[SIZE_CAL_WAVEFORM];

  void build_calibration_waveform();

  void arm();
  void checkAndUpdateState();
  //void getNextCalibrationValues(int val1, int val2);
  void dump_waveform();
  extern bool isRunning;
  void getNextCalOutputs(float &val1, float &val2);
  void startIfArmed();


}

#endif
