#ifndef INC_MODULATION
#define INC_MODULATION

//[](float a, float b) {
//            return (std::abs(a) < std::abs(b));
//        } // end of lambda expression
struct ModulationParams {
  unsigned int period;
  float amp;
  volatile unsigned int count;      //cycle counter
  float (*mod_func)(float t_frac);  //
  void update();
  void (*apply)(float val);
  float val;
};
void getModulationDacValues_free(float &dac0_val, float &dac1_val);
void getModulationDacValues_pumping(float &dac0_val, float &dac1_val);

const int N_MODULATIONS=9;
extern ModulationParams *GLB_mod_params[N_MODULATIONS];

void disableModulation();
void enableModulation();
void setModulationAmp(unsigned int idx, float amp);
void setModulationPeriod(unsigned int idx, unsigned int period);
void handleUpdateModulations();
void setDACOutput(int ax, int val);
void resetModulationPhase();
#endif
