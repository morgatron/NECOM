#include "steppers.h"
#include "pin_defs.h"

volatile int GLB_update_count = 0;

StepperWithParams steppers[5] = {
  { AccelStepper(AccelStepper::FULL4WIRE, PIN_STP_1.pin1, PIN_STP_1.pin2, PIN_STP_1.pin3, PIN_STP_1.pin4), 0, 0 }, // Could update these to use tuples or structs and put them in pin_defs.h
  { AccelStepper(AccelStepper::FULL4WIRE, PIN_STP_2.pin1, PIN_STP_2.pin2, PIN_STP_2.pin3, PIN_STP_2.pin4), 0, 0 },
  { AccelStepper(AccelStepper::FULL4WIRE, PIN_STP_3.pin1, PIN_STP_3.pin2, PIN_STP_3.pin3, PIN_STP_3.pin4), 0, 0 },
  { AccelStepper(AccelStepper::FULL4WIRE, PIN_STP_4.pin1, PIN_STP_4.pin2, PIN_STP_4.pin3, PIN_STP_4.pin4), 0, 0 },
  { AccelStepper(AccelStepper::DRIVER, TABLE_ROCK_PINS_H.stepPin, TABLE_ROCK_PINS_H.dirPin), 0, 0 }

};





void enableStepper(int idx) {
  //GLB_stoppedTimes[idx] = 0;
  steppers[idx]._stepper.enableOutputs();
  steppers[idx].stoppedTime = 0;
}

void move(int idx, int steps) {
  if (steps != 0) {
    AccelStepper *acc_stepper = &steppers[idx]._stepper;
    enableStepper(idx);

    steppers[idx].targetPos += steps;
    acc_stepper->moveTo(steppers[idx].targetPos);
  }
}

void offset(int idx, int new_offset) {
  //AccelStepper* acc_stepper = &steppers[idx]._stepper;
  //Serial.print("Offset "); Serial.print(idx); Serial.print(" ");
  //Serial.println(new_offset);
  StepperWithParams &swp = steppers[idx];
  if (new_offset != swp.offset) {
    enableStepper(idx);
    swp._stepper.moveTo(swp.targetPos + new_offset);
    swp.offset = new_offset;
  }
}

// Set max speed of all steppers
// e.g. max_speed 20
//
void set_max_speeds(float speed) {
  for (int k = 0; k < 4; k++) {
    steppers[k]._stepper.setMaxSpeed(speed);
  }
}
// Set acceleration of all steppers
// e.g. accel 110
//
void set_accels(float accel) {
  for (int k = 0; k < 4; k++) {
    steppers[k]._stepper.setAcceleration(accel);
  }
}



void updateSteppers() {
  // let them update themselves, and power them down when they've arrived.
  GLB_update_count++;
  for (int k = 0; k < 5; k++) {
    auto stp = &steppers[k];
    //AccelStepper *stpr = &(steppers[k]._stepper);
    //steppers[k]._stepper.run();
    //Serial.print("Motor "); Serial.print(k);
    //Serial.print(" | ");
    //Serial.println(GLB_stoppedTimes[k]);
    if (!stp->_stepper.isRunning() && stp->stoppedTime != -1) {
      //Serial.print("Motor "); Serial.print(k);
      //Serial.print(" is not running | ");
      //Serial.println(GLB_stoppedTimes[k]);

      //then we've recently stopped this motor
      int tNow = millis();
      if (stp->stoppedTime == 0)
        stp->stoppedTime = tNow;
      else if (tNow - stp->stoppedTime > 100) {
        stp->stoppedTime = -1;
        stp->_stepper.disableOutputs();
      }
    }
    stp->_stepper.run();
    //stp->_stepper.run();
    //stp->_stepper.run();
    //stp->_stepper.run();

  }
}
