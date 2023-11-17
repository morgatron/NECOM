
#include "Arduino.h"
namespace doAtTimes{

  typedef void(*funcPointer)();
  funcPointer* g_toDos;
  float* g_times;
  int g_N=0;
  int g_curIdx = 0;
  IntervalTimer g_timer;
  

  void _handleTrigger(){
    g_toDos[g_curIdx]();
    if(++g_curIdx >= g_N){ // if all todos done, finish
      g_timer.end();
      return;
    }
    else if ( (g_curIdx +1) < g_N ){
      g_timer.update(g_times[g_curIdx+1] - g_times[g_curIdx]);
    }
  }

  void doAtTimes(funcPointer toDos[], float times[], int N){ 
    g_curIdx = 0;
    g_times = times; // set the pointers to these- we need this memory not to be cleared until all todos are complete!
    g_toDos = toDos;
    g_N = N;
    g_timer.priority(16);
    //g_timer.begin(_handleTrigger, times[0]-micros()); 
    g_timer.begin(_handleTrigger, times[0]); 


    //Serial.println("retur");
    //return;
  
    
    //return;
    if(N>1){
      g_timer.update(times[1]-times[0]);
    }
  }
    
}
