/*
# Including nuclear-only modulation

For this, we will apply a field via the DAC, but only during the optical pumping phase of the first pulse in a sequence.



*/

#include "pin_defs.h"
#include "math.h"
#include "modulation.h"
#include "steppers.h"


float glb_doModulation = true;
float Bx1_val = 0;
float By1_val = 0;
int By_tot_mod = 0;
int Bx_tot_mod = 0;
int Ny_val = 0;
int Nx_val = 0;


auto fracSin = [](float frac) {
  return (float)sin(2 * M_PI * frac);
};
auto fracSquare = [](float frac) {
  return (float)( (frac < 0.5) ? 1.0 : -1.0 );
};


ModulationParams GLB_mod_Px = { 4096, 0, 0, fracSquare, [](float val) { // Maybe pump dir should use sin rather than square?
                                 offset(2, val);
                               } };
ModulationParams GLB_mod_Py = { 3072, 0, 0, fracSquare, [](float val) {
                                 offset(3, val);
                               } };
ModulationParams GLB_mod_Bx = { 47, 500, 0, fracSin, [](float val) {
                                 //setDACOutput(0, val + Bx1_val);
                                 Bx_tot_mod = val + Bx1_val;
                               } };
ModulationParams GLB_mod_By = { 37, 500, 0, fracSin, [](float val) {
                                 //setDACOutput(1, val + By1_val);
                                 By_tot_mod = val + By1_val;
                               } };
ModulationParams GLB_mod_Bx_1 = { 1231, 000, 0, fracSquare, [](float val) {
                                   Bx1_val = val;
                                 } };
ModulationParams GLB_mod_By_1 = { 1082, 000, 0, fracSquare, [](float val) {
                                   By1_val = val;
                                 } };
ModulationParams GLB_mod_Nx = { 1231, 0, 0, fracSin, [](float val) {
                                   Nx_val = val;
                                 } };
ModulationParams GLB_mod_Ny = { 1082, 0, 0, fracSin, [](float val) {
                                   Ny_val = val;
                                 } };
ModulationParams GLB_mod_TableH = { 2560, 0, 0, fracSquare, [](float val) {
                                   offset(4, val);
                                 } };
                      
ModulationParams *GLB_mod_params[N_MODULATIONS] = {
  &GLB_mod_Px, &GLB_mod_Py,
  &GLB_mod_Bx, &GLB_mod_By,
  &GLB_mod_Bx_1, &GLB_mod_By_1,
  &GLB_mod_Nx, &GLB_mod_Ny,
  &GLB_mod_TableH,
};

void getModulationDacValues_free(float &dac0_val, float &dac1_val){
  dac0_val = Bx_tot_mod;
  dac1_val = By_tot_mod;  
}

void getModulationDacValues_pumping(float &dac0_val, float &dac1_val){
  dac0_val = Bx_tot_mod + Nx_val;
  dac1_val = By_tot_mod + Ny_val;  
}


void ModulationParams::update(){
  this->count++;
  if (this->count >= this->period){
    this->count = 0;
  }
  this->val = this->amp * this->mod_func((float)this->count / this->period);
  this->apply(this->val);
}
//called on interrupt from timer
void handleUpdateModulations() {
  if( !glb_doModulation){
    return;
  }
  //Serial.print("Bx1_val, By1_val:"); Serial.print(Bx1_val); Serial.print(", "); Serial.println(By1_val);
  static int mod_count = 0;
  //Serial.print("mod_count: "); Serial.println(mod_count);
  for (auto params : GLB_mod_params) {
    params->update();
    // params->count++;
    // if (params->count >= params->period)
    //   params->count = 0;
    // auto new_val = params->amp * params->mod_func((float)params->count / params->period);
    // params->apply(new_val);
  }
  //for (int k =0; k <4; k++)
  mod_count++;

  /*
  GLB_mod.count++;

  int new_offs = GLB_mod.amp* sin( 2*M_PI*GLB_mod.count / (float) GLB_mod.period);
  int new_offs = GLB_mod.amp*GLB_mod.mod_func((float)GLB_mod.count/GLB_mod.period);
  GLB_mod.apply(new_offs);
  if ( GLB_mod.count >= GLB_mod.period){
     GLB_mod.count = 0;
    // update target positions
  }

  offset(0, new_offs);
  */
}

void setModulationPeriod(unsigned int idx,  unsigned int period){
  ModulationParams *params = GLB_mod_params[idx];
  params->period = period;
}

void setModulationAmp(unsigned int idx, float amp){

  ModulationParams *params = GLB_mod_params[idx];
  params->amp = amp;  
  // // Engage interupts if some amps are non-zero.
  // bool bAllAmpsZero = true;
  // for (int k = 0; k < 4; k++) {
  //   if (GLB_mod_params[k]->amp) {
  //     bAllAmpsZero = false;
  //   }
  // }
  
}


/*
void enableExternalModulation() {
  detachInterrupt(digitalPinToInterrupt(PIN_MODULATION_CLOCK_B));
  attachInterrupt(digitalPinToInterrupt(PIN_MODULATION_CLOCK_A), handleUpdateModulations, RISING);
  return;
}
*/
void enableModulation() {
  glb_doModulation = true;
}
void disableModulation() {
  glb_doModulation = false;
  // This does not set outputs back to zero, but leaves them in their last state.
}
void resetModulationPhase(){
  disableModulation();
  for (auto params : GLB_mod_params){
    params->count=0;
  }
  enableModulation();
}
