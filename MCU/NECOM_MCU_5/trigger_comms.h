#include "pin_defs.h"
namespace triggerComms {
  int triggersDone = 0;
  int glb_triggerCommsInterval = 100; //us
  IntervalTimer triggerCommsTimer;
  const int N_SIGNALS = 10; //N_MODULATIONS + 2
  bool signals[N_SIGNALS];

  void _handleTriggerComms(){ 
    static unsigned int curIndex=0;
    int newState;
    if(curIndex>=N_SIGNALS){ // Finished sending data
      triggerCommsTimer.end();
      newState = false;
      curIndex = 0; //reset for next run
    }
    else{
      newState = signals[curIndex];
      triggersDone++;
      curIndex++;
    }
    digitalWrite(PIN_SYNC_DATA_OUTPUT, newState);
  }

  void _startTriggerCommsHandler(){ 
    _handleTriggerComms();
    triggerCommsTimer.begin(_handleTriggerComms, glb_triggerCommsInterval);  
  }
  void startTriggerCommsAt(int startTimeUs){
    triggerCommsTimer.begin(_startTriggerCommsHandler, startTimeUs);
    triggerCommsTimer.priority(1);
  }

}
