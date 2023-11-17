#include <SerialCommands.h>

void cmd_enable_modulation(SerialCommands *sender);
void cmd_disable_modulation(SerialCommands *sender);
void cmd_set_modulation(SerialCommands *sender);
void cmd_set_max_speeds(SerialCommands *sender);
void cmd_set_accels(SerialCommands *sender);
void cmd_move(SerialCommands *sender);
void cmd_set_dac(SerialCommands *sender);
void cmd_enable_pps_lock(SerialCommands *sender);
void cmd_disable_pps_lock(SerialCommands *sender);
void cmd_minute_sync(SerialCommands *sender);
void cmd_start_calibration(SerialCommands *sender);
void cmd_dump_cal_waveform(SerialCommands *sender);
void cmd_set_rep_period_us(SerialCommands *sender);
void cmd_reset_modulation_phase();


void cmd_unrecognized(SerialCommands *sender, const char *cmd);
