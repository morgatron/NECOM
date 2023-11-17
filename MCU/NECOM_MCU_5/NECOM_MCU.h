#ifndef _INC_NECOM_MCU
#define _INC_NECOM_MCU

void enablePPSLock();
void disablePPSLock();
void handlePPSArrival();
void triggerCalibration();
void setDACOutputs(int val1, int val2);
void setRepPeriodManual(float repRate);

extern volatile unsigned int glb_PpsArrivalsSinceMinute;

#endif
