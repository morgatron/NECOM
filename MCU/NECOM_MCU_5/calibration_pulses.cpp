#include "calibration_pulses.h"
#include "Arduino.h"
#include "pin_defs.h"

/*
Implementation plan:
* Change the updateModulations code to update a global value to be written to the DAC, not update the DAC directly
* When triggered externally (i.e. via computer), the MCU will begin calibration sequence mode at the start of the next minute
* During the calibration sequence, the DAC values to be uploaded will be set to the next value in the calibration waveform
* Generally the value to be set will be calculated at the end of handleTick. New values will be written at the start.

* if(calibrating){
    analogWrite(DAC0, nextOutputDac0)
    analogWrite(DAC1, nextOutputDac1)

  
  //at end
  if(calibrating){
  nextOutputDac0 = calibration::updateState()
  
  }
  else nextOutputDac0 = modulation::nextVal

}
*/

namespace calState{
  volatile bool armed = false;
  volatile int currentSequence = 0;
  volatile int wvfmIdx = 0;
  int wvfmLen = 0;
}

namespace calibration{

int16_t calibration_waveform[SIZE_CAL_WAVEFORM];


bool isRunning = false;

void dump_waveform(){
  Serial.print("wvfm: "); Serial.print(SIZE_CAL_WAVEFORM); Serial.print(" "); 
  Serial.write((char*)calibration_waveform, 2*SIZE_CAL_WAVEFORM );
  Serial.println(":END");
}

void arm(){
  calState::armed = true;
}

void begin(){
  calState::armed = false;
  isRunning=true;
  digitalWrite(PIN_CALIBRATION_INDICATOR, true);
  calState::currentSequence = 0;
  calState::wvfmIdx = 0;
}

int V(float V){
  int val = 2048 + V/3.3*4096;
  if(val>4095){
    val=4095;
  }
  else if (val<0){
    val=0;
  }
  return val;
} 

float V2ptX = 65; // very rough!
float V2ptY = 96; // very fough!



const size_t N_SINS = 20;
const float sin_pattern[N_SINS][2] = {
    {0.5, 4},
    {0.75,4},
    {1, 4},
    {1.25, 4},
    {1.5,4},
    {2,4},
    {3,3},
    {4,2},
    {5,2},
    {6,2},
    {10,2},
    {35,1},
    {55, 0.6},
    {70, 0.4},
    {80, 0.2},
    {90, 0.2},
    {110,0.2},
    {130, 0.2},
    {160, 0.1},
    {190, 0.1},
};

void make_sin_pattern_waveform(const float pattern[][2], int16_t wvfm[], const size_t N, const int amp, float &tEnd){
  size_t idx = 0;
  float tEndOfLast = 0;
  float t= 0;
  //for(float pair[2] : pattern){
  for(size_t i=0; i< N_SINS; i++){
    const float *pair = pattern[i];
    float freq = pair[0];
    float duration = pair[1];
    float finTime= tEndOfLast + duration;
    while(t<finTime){
      if(idx>N){
        Serial.println("OVERFLOW IN MAKE SIN PATTERN!!!!!!!!!!!!");
      }
      wvfm[idx] = sin(2*M_PI*freq*(t-tEndOfLast))*amp;// + ZERO;
      t = ++idx/SAMPLE_RATE;
    }
    tEndOfLast = t;
  }
 
  tEnd = t;
}

//const uint16_t ZERO = pow(2,16)/2;

void make_square_waveform(int16_t *wvfm, const size_t N, float pulse_duration, int amp, float& tFinish){
  unsigned int idx =0;
  float t = 0;
  while(t<16){
    if(idx>N){
      Serial.println("OVERFLOW IN MAKE SQUARE WAVEFORM!!!!!!!!!!!!");
    }
    wvfm[idx] = amp;// + ZERO;
    t = ++idx/SAMPLE_RATE;
  }
}
// Waveform will be use up the full range- will need to be divided down to calibrate
void build_calibration_waveform(){
  int finishIdx=0;
  for(int k=0; k< SIZE_CAL_WAVEFORM; k++){
    calibration_waveform[k] = 0;//ZERO;
  }
  float tFinish=0; // not used
  float tStart = 4;//seconds
  int startIdx = (int)(tStart*SAMPLE_RATE);
  make_sin_pattern_waveform(sin_pattern, &calibration_waveform[ startIdx ], SIZE_CAL_WAVEFORM-startIdx, (pow(2,15)-1)/2, tFinish);
  float square_start_time = 4 + 38 + 10; // Sin waveform should take 38s
  int square_start_idx = square_start_time*SAMPLE_RATE; 
  float square_wave_duration = 16;//s
  make_square_waveform(&calibration_waveform[square_start_idx], SIZE_CAL_WAVEFORM-square_start_idx, square_wave_duration, pow(2,15)-1, tFinish);
  //finishIdx = tFinish*SAMPLE_RATE + square_start_idx;
  calState::wvfmLen = 80*SAMPLE_RATE;//finishIdx+1;
  Serial.print("waveform length: "); Serial.println(calState::wvfmLen);
}

float bxCal = 2000.0/pow(2,15);
float bzCal = 1200/pow(2,15);

void getNextCalOutputs(float &val1, float &val2){
 if(isRunning){
    float nextVal = calibration_waveform[calState::wvfmIdx];
    if (calState::currentSequence==0){
      val1 = nextVal*bzCal;
      val2 = 0;
    }
    else{
      val1 = 0;
      val2 = nextVal*bxCal;
    }

    // 
    if( (++calState::wvfmIdx) >= calState::wvfmLen){
      calState::wvfmIdx=0;
      if(calState::currentSequence == 1){ // If we've just finished the second sequence, reset cal state
        calState::currentSequence=0;
        isRunning = false;
        digitalWrite(PIN_CALIBRATION_INDICATOR, false);

      }
      else{ // Otherwise move onto the next sequence.
        calState::currentSequence = 1;
      }
    }
  }
}

void startIfArmed(){
  if (calState::armed){ // IF armed, then we'll start the sqeuence
    begin();
  }
}


}// namespace calibration
