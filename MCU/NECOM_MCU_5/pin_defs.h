#ifndef _INC_PIN_DEFS
#define _INC_PIN_DEFS
#include "Arduino.h"

// Pins
const int PIN_SYNC_TRIG_OUTPUT2 = 24; // Synchronisation with other instruments
const int PIN_SYNC_TRIG_OUTPUT = 32; // Synchronisation with other instruments
const int PIN_SYNC_DATA_OUTPUT = 31; // communicating modulation/timing state to computer code.
const int PIN_SYNC_MINUTE_OUTPUT = 30; // Minute output for DMT box

const int PIN_CALIBRATION_INDICATOR = 12;



const int PIN_PPS_INPUT = 28;
const int DAC0 = A21; // To coils
const int DAC1 = A22;
const int ledPin = LED_BUILTIN;  // the pin with a LED

struct FourPinGroup{
  int pin1;
  int pin2;
  int pin3;
  int pin4;
};

const FourPinGroup PIN_STP_1 = {2,4,3,5}; 
const FourPinGroup PIN_STP_2 = {8, 10, 9, 11};
const FourPinGroup PIN_STP_3 = { 23, 21, 22, 20};
const FourPinGroup PIN_STP_4 = {17, 15, 16, 14};

struct StepperDrvPins{
  int stepPin;
  int dirPin;
  int sleepPin;
};

const StepperDrvPins TABLE_ROCK_PINS_H = {27, 26, 25};

#endif
