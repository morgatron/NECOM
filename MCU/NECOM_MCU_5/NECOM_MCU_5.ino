/* Control two sets of mirrors to align/modulate pump... 
 * AND control low and high speed magnetic modulations
 * AND provide clock for the experiment (by triggering FGs)
 * AND trigger the scope
 * AND digitally encode sync information to the scope
 * AND synchronise with external pulse-per-second
 * 
 * Commands to write:
 * - mv (motor) (number of steps)
 * - offs (motor) (number of steps) //offset from current nominal position, for modulation purposes
 * - offs (no command) <- set offsets to zero
 * - mod_state (which) (true/false)
 * - step_rate (steps per second)
 * - set_mod_amp (which) (period) (A amp) (B amp) (C amp) (D amp)
 *
 *
 * How does modulation work?
 *  When a modulation is set, it is associated with a certain input pin. Every time this pin has a leading edge, the next part of the modulation is triggered.
 * The 'period' parameter indicates how many triggers for a full modulation cycle. _probably_ should be an even number, or modulations will be weird shaped.
 * 
 * How dows moving work?
 * - "mv" should set the 'home' position
 * - "offs" should move relative to the home position.
 * 
 * To do this:
 * - "move" should moveTo (targetPosition + steps)
 * - "offs" moveTo (targetPosition + steps -prev_offs)
 * 
 * Alternatively:
 * - keep track of offsets and central positions in a global array
 * - update the global arrays on commands, and use moveTo appropriately (moveTo(


2023 July:
Now including calibration pulses etc. Reading how the
 * ))
 */

#include <SerialCommands.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include "modulation.h"
#include "steppers.h"
#include "pin_defs.h"
#include "trigger_comms.h"
#include "cmds.h"
#include "NECOM_MCU.h"
#include "calibration_pulses.h"
#include "do_at_times.h"

IntervalTimer mainTimer;





// ----- PARAMS ---------------------------------

// Timing
const float REP_RATE = 256*1.000025; // clocks eems to be ~25ppm out
const float NOMINAL_TICK_PERIOD = 1000000/REP_RATE; // Nominal period in us
// Steppers
const int GLB_ACCELERATION = 8000;
const int GLB_MAX_SPEED = 4000;
const int GLB_MAX_SPEED_TABLE = 3000;
const int GLB_ACCELERATION_TABLE = 2000;


// Global state keeping
int glb_lastTickTime = 0;
volatile float glb_PpsArrivalErr = 0;// deviation from ideal PPS arrival time. Just for logging purposes
volatile bool glb_bPpsJustArrived = false;
volatile unsigned int glb_PpsArrivalsSinceMinute = 0;
volatile unsigned int glb_timeStamp = 0;

float glb_mainTimerPeriod = NOMINAL_TICK_PERIOD; // this will be updated to match the external PPS.


float glb_curDacOut_0 = 0;
float glb_curDacOut_1 = 0;

float glb_curDacOut_0_pump = 0;
float glb_curDacOut_1_pump = 0;

void setDACOutputs(int val1, int val2){
  analogWriteDAC0( val1 + 2047);
  analogWriteDAC1( val2 + 2047);
  //Serial.print(val1); Serial.print(" | "); Serial.println(val2);
}

void setDACsMain(){
  setDACOutputs(glb_curDacOut_0, glb_curDacOut_1);
  digitalWrite(6,true);
}

void setDACsForPumping(){
  setDACOutputs(glb_curDacOut_0_pump, glb_curDacOut_1_pump);
  digitalWrite(6,false);
}

/* A tick is the thing that happens at the start of every measurement period
 * Things that need doing:
 * - Trigger function generators (ie. start all external hardware)
 * - update all modulations to new values
 * - send synchronisation information, including: 
      - whether a PPS has arrive since the last tick
      - whether a minute boundary has passed
      - the phase of the modulations
   - The synchronisation information takes the form of a stream of TTLs 
*/

doAtTimes::funcPointer toDos[] = { [](){setDACsForPumping();}, 
              [](){setDACsMain();}, 
              [](){setDACsForPumping();}, 
              [](){setDACsMain();} };
float times[4];
void handleTick(){
  glb_lastTickTime = micros();
  digitalWrite(PIN_SYNC_TRIG_OUTPUT2, true); 
  digitalWrite(PIN_SYNC_TRIG_OUTPUT, true);
  //setDACOutputs(glb_curDacOut_0, glb_curDacOut_1);

  // times[0] = glb_lastTickTime+50;
  // times[1] = glb_lastTickTime+180; 
  // times[2] = glb_lastTickTime+glb_mainTimerPeriod/2+50;
  // times[3] = glb_lastTickTime+glb_mainTimerPeriod/2+180;
  times[1] = 100; 
  times[2] = glb_mainTimerPeriod/2+10;
  times[3] = glb_mainTimerPeriod/2+100;
  times[0] = 20-(micros()-glb_lastTickTime);

  //times[0] = glb_lastTickTime + 50;
  //times[1] = glb_lastTickTime + 190;
  doAtTimes::doAtTimes(toDos, times, 2);
  //doAtTimes::doAtTimes(toDos, times, 4);

  //doAtTimes([](){} )
  //Is it possible to trigger this from DMA?
  //Timing may be particularly important for bumping the Xe
  //as we don't want th signal still on after the pump is off.
  //The timing consistency is maybe even more important.
  //Ideally the DACs are written to in the previous step, then
  //At this point the outgoing TTL acts as a hardware trigger for the DACs
  //Similalry for going high and low, so it syncs with the
  //setDACOutputs(glb_curDacOut_0, glb_curDacOut_1);

  if(calibration::isRunning){
    calibration::getNextCalOutputs(glb_curDacOut_0, glb_curDacOut_1); 
  }
  else{
    handleUpdateModulations();  //update DAC outputs. Update targets for stepper
    getModulationDacValues_free(glb_curDacOut_0, glb_curDacOut_1);
    getModulationDacValues_pumping(glb_curDacOut_0_pump, glb_curDacOut_1_pump);
    updateSteppers();  // continue moving steppers toward wherever they're supposed to go.
  }




  //Make sure the pulse stays high for at least 200 us.
  int tElapsed = micros() - glb_lastTickTime;
  if (tElapsed < 200){
    delayMicroseconds(200-tElapsed);
  }
  digitalWrite(PIN_SYNC_TRIG_OUTPUT2, false);
  digitalWrite(PIN_SYNC_TRIG_OUTPUT, false);
  setDACOutputs(glb_curDacOut_0, glb_curDacOut_1);


  
  triggerComms::signals[0] = glb_PpsArrivalsSinceMinute == 0; // Minute just past
  triggerComms::signals[1] = glb_bPpsJustArrived;
  for( int k =0; k<N_MODULATIONS; k++){
    ModulationParams *modParams = GLB_mod_params[k];
    triggerComms::signals[k+2] = ((float)modParams->count / modParams->period) < 0.5; // High if in the first half of the cycle
  }
  triggerComms::startTriggerCommsAt(glb_lastTickTime +1000 -micros());
  
  glb_bPpsJustArrived = false;
}

void enablePPSLock(){
    attachInterrupt(digitalPinToInterrupt(PIN_PPS_INPUT), handlePPSArrival, RISING);
}
void disablePPSLock(){
    detachInterrupt(digitalPinToInterrupt(PIN_PPS_INPUT));
}


void setRepPeriodManual(float repPeriod){
  disablePPSLock();
  glb_mainTimerPeriod = repPeriod;
  mainTimer.update(repPeriod);

}
// handle what happens when a PPS signal comes in
// Main task is to update the mainTimer tick rate to match the PPS arrivals
volatile int timer_resets=0;
volatile int times_armed =0;
void handlePPSArrival(){
  //static int arrivalErrLast = 0; 
  // update the mainTimer to keep this locked.
  // We're expecting this ~500us trhough a trace
  int tNow = micros(); 
  calibration::startIfArmed();

  glb_bPpsJustArrived = true;
  if(++glb_PpsArrivalsSinceMinute == 60){
    digitalWrite(PIN_SYNC_MINUTE_OUTPUT, true);
    glb_PpsArrivalsSinceMinute = 0; // should really probably do the modulo trick
  }
  else if(glb_PpsArrivalsSinceMinute==1){
    digitalWrite(PIN_SYNC_MINUTE_OUTPUT, false);    
  }
  
  //float arrivalErr = (10* glb_PpsArrivalErr + (NOMINAL_TICK_PERIOD/2 - (tNow - glb_lastTickTime)))/11; // Time since the last tick
  float arrivalErr = 3*NOMINAL_TICK_PERIOD/4 - (tNow - glb_lastTickTime); // Time since the last tick

  if (glb_PpsArrivalErr){ // Don't try to lock on the first PPS arrival
    float cumErr = (arrivalErr + glb_PpsArrivalErr);
    float diffErr = arrivalErr - glb_PpsArrivalErr;
    float speedUpFactor = (abs(arrivalErr) > 20.0) ? 20 : 1; 
    float periodAdjust = cumErr / REP_RATE / 200 * speedUpFactor;

    periodAdjust  += diffErr/REP_RATE/1.2;
    glb_mainTimerPeriod -= periodAdjust;
    mainTimer.update(glb_mainTimerPeriod);
  }
  glb_PpsArrivalErr = arrivalErr;

 

}



char serial_command_buffer_[64];
SerialCommands serial_commands_(&Serial, serial_command_buffer_, sizeof(serial_command_buffer_), "\r\n", " ");

SerialCommand cmd_func_list[] = {
  { "mv", cmd_move },
  { "accels", cmd_set_accels },
  { "max_speeds", cmd_set_max_speeds },
  { "modOn", cmd_enable_modulation },
  { "modOff", cmd_disable_modulation },
  { "set_mod", cmd_set_modulation },
  { "set_dac", cmd_set_dac },
  { "PPS_lock_on", cmd_enable_pps_lock },
  { "PPS_lock_off", cmd_disable_pps_lock },
  { "minute_sync", cmd_minute_sync },
  { "arm_cal", cmd_start_calibration},
  { "dump_cal", cmd_dump_cal_waveform},
  { "set_period", cmd_set_rep_period_us},
  { "reset_mod_phase", cmd_reset_modulation_phase},


};


void setup() {
  analogWriteResolution(12);
  set_max_speeds(GLB_MAX_SPEED);
  set_accels(GLB_ACCELERATION);
  pinMode(ledPin, OUTPUT);
  pinMode(PIN_PPS_INPUT , INPUT);
  pinMode(PIN_SYNC_TRIG_OUTPUT, OUTPUT);
  pinMode(PIN_SYNC_TRIG_OUTPUT2, OUTPUT);

  pinMode(PIN_SYNC_DATA_OUTPUT, OUTPUT);
  pinMode(PIN_SYNC_MINUTE_OUTPUT, OUTPUT);
  pinMode(PIN_CALIBRATION_INDICATOR, OUTPUT);
  //pinMode(PIN_TRIGGER_CALIBRATION_2, OUTPUT);


  pinMode(DAC0, OUTPUT);
  pinMode(DAC1, OUTPUT);

  pinMode(6, OUTPUT);
  mainTimer.begin(handleTick, 1000000.0/REP_RATE); 
  mainTimer.priority(32);
  Serial.begin(9600);


  serial_commands_.SetDefaultHandler(&cmd_unrecognized);
  for (auto &cmd : cmd_func_list) {
    Serial.println(cmd.command);
    serial_commands_.AddCommand(&cmd);
  }

  disableModulation();
  enablePPSLock();
  calibration::build_calibration_waveform();
  //disablePPSLock();


  //steppers setup
  steppers[4]._stepper.setEnablePin(TABLE_ROCK_PINS_H.sleepPin);
  steppers[4]._stepper.setMaxSpeed(GLB_MAX_SPEED_TABLE);
  steppers[4]._stepper.setAcceleration(GLB_ACCELERATION_TABLE);
  pinMode(TABLE_ROCK_PINS_H.sleepPin, OUTPUT);
  //pinMode(TABLE_ROCK_PINS_H.stepPin, OUTPUT);
  //pinMode(TABLE_ROCK_PINS_H.dirPin, OUTPUT);
  digitalWrite(TABLE_ROCK_PINS_H.sleepPin, HIGH);


}



void loop() {
  static int tLoop = 0;
  static int loop_counter = 0;
  static int last_pps_val = 0;
  int tNow = millis();
  int tElapsed = tNow - tLoop;
  if (glb_PpsArrivalsSinceMinute != last_pps_val)  //print status once in a while
  {
    last_pps_val = glb_PpsArrivalsSinceMinute; 
    Serial.print("\r\n");
    Serial.print("PPS arrivals since last minute: ");
    Serial.println(glb_PpsArrivalsSinceMinute);

    Serial.println("move plus");
    loop_counter++;
    Serial.print("loop time: ");
    Serial.println((float)tElapsed / loop_counter);
    loop_counter = 0;
    tLoop = tNow;

    
    Serial.print("calRunning: ");
    Serial.println(calibration::isRunning);
 
    Serial.print("resets: "); Serial.println(timer_resets);
    Serial.print("times armed: "); Serial.println(times_armed);

    Serial.println();
    
    

    /*
    Serial.println(steppers[0]._stepper.speed());
    Serial.println(steppers[1]._stepper.speed());
    Serial.println(steppers[2]._stepper.speed());
    Serial.println(steppers[3]._stepper.speed());
    Serial.println("");
    Serial.print("count: ");
    Serial.println(GLB_update_count);
    Serial.println();
    */
    for(int k=0; k<6; k++){
      Serial.print(triggerComms::signals[k]);
      Serial.print("|");
    }



    //Print sync timer status
    Serial.print("main timer period: "); Serial.println(glb_mainTimerPeriod);
    Serial.print("PPS Arrival error: "); Serial.println(glb_PpsArrivalErr);
    float fracTimerDeviation = (float)abs(NOMINAL_TICK_PERIOD - glb_mainTimerPeriod)/NOMINAL_TICK_PERIOD;
    if ( fracTimerDeviation >0.05 ){
      Serial.print("WARNING: SEEM TO BE LOCKED WAY OUTSIDE THE NOMINAL TIMING. FRACTIONAL ERROR IS "); Serial.println(fracTimerDeviation);
    }


  }
  //Serial.println(arr[30000]);

  //updateSteppers();
  //updateSteppers();
  serial_commands_.ReadSerial();
}
