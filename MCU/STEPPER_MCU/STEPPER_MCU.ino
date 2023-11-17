/*

MCU _just_ to drive the stepper. It will be programmed over USB. 
It will have a single communication input pin. A rising edge makes it being travelling to a target position. 
A falling edge makes it begin moving back.

It will trigger sleep mode when it's been stationary for a short time to avoid over-heating the motor.
*/

#include <AccelStepper.h>


int PIN_MOVEMENT_TRIGGER = D2;

// Define a stepper and the pins it will use
int PIN_SLEEP = D5;
int PIN_STEP = D6;
int PIN_DIR = D7;
AccelStepper stepper(AccelStepper::DRIVER, PIN_STEP, PIN_DIR); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5
//int chans[] = {D0,D1,D2,D3,D4,D5};

void setup()
{  
  // Change these to suit your stepper if you want
  //stepper.setMaxSpeed(100);
  //stepper.setAcceleration(20);
  //stepper.moveTo(50);
  //for
  //digitalWrite(14, true);
  for(int k=0; k<6; k++){
    pinMode(chans[k], OUTPUT);
  }
  Serial.begin(9600);
  cmds::setup();
  stepper.setSpeed(10000);
  stepper.runSpeed();
 
}
//int ans[] = {5,4,0,2,14,12};

void loop()
{
  moveSteppers();
  
    // If at the end of travel go to the other end
    //if ( (stepper.distanceToGo() == 0) && false ) 
    //  stepper.moveTo(-stepper.currentPosition());

    //stepper.run();
    for(int k=0; k<6; k++){
      Serial.print(chans[k]); Serial.print(" | ");
      digitalWrite(chans[k], 1);
      delay(600);
      digitalWrite(chans[k], 0);
    }
    Serial.println();
    //      delay(500);

    //delay(100);

}
