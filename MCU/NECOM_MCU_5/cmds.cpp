#include "cmds.h"
#include "modulation.h"
#include "steppers.h"
#include "NECOM_MCU.h"
#include "calibration_pulses.h"


void cmd_enable_modulation(SerialCommands *sender) {
  enableModulation();
}

void cmd_disable_modulation(SerialCommands *sender) {
  disableModulation();
}
void cmd_enable_pps_lock(SerialCommands *sender){
  enablePPSLock();  
}
void cmd_disable_pps_lock(SerialCommands *sender){
  disablePPSLock();
}

// mod (mod_index) (amplitude) (period)
// mod 1 100 10
void cmd_set_modulation(SerialCommands *sender) {
  char *cmdStr;
  char *endptr = NULL;
  if (!(cmdStr = sender->Next())) {
    Serial.println("syntax: mod <axIndex> <amplitude> <period>");
    return;
  }
  unsigned int idx;
  if (strcmp("0", cmdStr) == 0)
    idx = 0;
  else if (strcmp("1", cmdStr) == 0)
    idx = 1;
  else if (strcmp("dac0", cmdStr) == 0)
    idx = 2;
  else if (strcmp("dac1", cmdStr) == 0)
    idx = 3;
  else if (strcmp("dac0_1", cmdStr) == 0)
    idx = 4;
  else if (strcmp("dac1_1", cmdStr) == 0)
    idx = 5;
 else if (strcmp("n_a", cmdStr) == 0)
    idx = 6;
  else if (strcmp("n_b", cmdStr) == 0)
    idx = 7;
  else if (strcmp("table_h", cmdStr) == 0)
    idx = 8;
  else {
    Serial.print("err: Can't understand modulation index: ");
    Serial.println(cmdStr);
    return;
  }

 
  float ampl = strtof(sender->Next(), &endptr);
  if (endptr != cmdStr) {
    setModulationAmp(idx, ampl);
  }
  unsigned int period = strtol(sender->Next(), &endptr, 10);
  if (endptr != cmdStr) {
    setModulationPeriod(idx, period);
  }
}


void cmd_set_max_speeds(SerialCommands *sender) {
  const char *cmdStr = sender->Next();
  const float paramVal = atoi(cmdStr);
  if (paramVal == 0) {
    if (strcmp(cmdStr, "0") != 0) {
      Serial.print("err: Don't understand ");
      Serial.println(cmdStr);
      return;
    }
  }
  set_max_speeds(paramVal);
}


void cmd_set_accels(SerialCommands *sender) {
  const char *cmdStr = sender->Next();
  const float paramVal = atoi(cmdStr);
  if (paramVal == 0) {
    if (strcmp(cmdStr, "0") != 0) {
      Serial.print("err: Don't understand ");
      Serial.println(cmdStr);
      return;
    }
  }
  set_accels(paramVal);
}

void cmd_move(SerialCommands *sender) {
  // Expect commands of axis/distance pairs.
  //  E.g. mv B10 A2000 for moving axis B by 10 and axis A by 2000
  //  For each pair, this will:
  //  calculate the index based on alphabetic axis letter
  //  set the appropriate move position, probably just based on where it currently is
  //  write a sensible error and stop if something makes no sense

  auto serial = sender->GetSerial();
  Serial.println("info: Running move command");
  const char *cmdStr;
  while ((cmdStr = sender->Next())) {
    Serial.print("cmd: ");
    Serial.println(cmdStr);
    char axChar = tolower(cmdStr[0]);
    unsigned int axIndex = (int)axChar - (int)'a';
    if (axIndex > 4) {
      serial->print("err: Can't interpret ");
      serial->println(cmdStr);
      break;
    }

    const char *stepStr = &cmdStr[1];
    int steps = atoi(stepStr);
    Serial.print(axIndex);
    Serial.print(" : ");
    Serial.println(steps);
    if (steps == 0) {
      if (strcmp(stepStr, "0") != 0) {
        serial->print("err: Don't understand ");
        serial->println(stepStr);
      }
    }
    move(axIndex, steps);
  }
}


void cmd_set_dac(SerialCommands *sender) {
  auto serial = sender->GetSerial();
  Serial.println("info: Running set_dac command");
  Serial.println("CURRENTLY SET_DAC ISN'T WORKING!");
  return;
  /*
  const char *cmdStr;
  while ((cmdStr = sender->Next())) {
    Serial.print("cmd: ");
    Serial.println(cmdStr[0]);
    char axChar = tolower(cmdStr[0]);
    Serial.println(axChar);

    int axIndex;
    if (axChar == 'x') {
      axIndex = 0;
    } else if (axChar == 'y') {
      axIndex = 1;
    } else {
      Serial.println("Err: dac output axis must be X or Y");
      return;
    }

    const char *valStr = &cmdStr[1];
    int val = atoi(valStr);
    Serial.print(axIndex);
    Serial.print(" : ");
    Serial.println(val);
    if (val == 0) {
      if (strcmp(valStr, "0") != 0) {
        serial->print("err: Don't understand ");
        serial->println(valStr);
      }
    }
    setDACOutput(axIndex, val);
  }
  */
}

//should be called ~500ms before the clock clicks over to 1 minute
void cmd_minute_sync(SerialCommands *sender){
  char *cmdStr;
  int seconds = 0;
  if ( (cmdStr = sender->Next()) ){
    seconds = atoi(cmdStr);
  }
  glb_PpsArrivalsSinceMinute = seconds; // so that next PPS is a minute
  Serial.println("minute synced");
}

// Will trigger calibration procedure at next PPS
void cmd_start_calibration(SerialCommands *sender){
  calibration::arm();
}

void cmd_dump_cal_waveform(SerialCommands *sender){
  calibration::dump_waveform();
}
void cmd_set_rep_period_us(SerialCommands *sender){
  float repPeriod = 0;
  char *cmdStr;
  if ( (cmdStr = sender->Next()) ){
    repPeriod = atof(cmdStr); 
    if (repPeriod){
      setRepPeriodManual(repPeriod);
      Serial.print("setting period to "); Serial.println(repPeriod);
    }
    else{
      Serial.print("Couldn't set period to '"); Serial.print(cmdStr); Serial.println("'");
    }
  }
  else{
    Serial.println("Couldn't set period, none given");
  }

 
}
void cmd_reset_modulation_phase(){
  resetModulationPhase();
}


void cmd_unrecognized(SerialCommands *sender, const char *cmd) {
  auto serial = sender->GetSerial();
  serial->print("err: Unrecognized command [");
  serial->print(cmd);
  serial->println("]");
}
