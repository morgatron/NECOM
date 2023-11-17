#ifndef _INC_STEPPERS
#define _INC_STEPPER

#include <AccelStepper.h>

struct StepperWithParams {
  AccelStepper _stepper;
  float offset;
  int targetPos;
  volatile int stoppedTime;
};
//volatile int GLB_stoppedTimes[4] = { 0, 0, 0, 0 };  // Store how long each motor has been idle. Kept seperate as it needs to be volatile... but maybe it doesn't need to be??
extern StepperWithParams steppers[5];

extern volatile int GLB_update_count;

void enableStepper(int idx);
void move(int idx, int steps);
void offset(int idx, int new_offset);
void set_max_speeds(float speed);
void set_accels(float accel);
void updateSteppers();

#endif
