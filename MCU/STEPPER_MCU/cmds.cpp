#include "cmds.h"
namespace cmds{
  //COMMAND FUNCTIONS
  void cmd_move(SerialCommands *sender, const char *cmd);
  void cmd_set_accel(SerialCommands *sender, const char *cmd);
  void cmd_go_period(SerialCommands *sender, const char *cmd);
  void cmd_set_max_speed(SerialCommands *sender, const char *cmd);
  void cmd_set_amp(SerialCommands *sender, const char *cmd);
  void cmd_stop(SerialCommands *sender, const char *cmd);

  void cmd_unrecognized(SerialCommands *sender, const char *cmd);


  char serial_command_buffer_[64];
  SerialCommands serial_commands_(&Serial, serial_command_buffer_, sizeof(serial_command_buffer_), "\r\n", " ");

  SerialCommand cmd_func_list[] = {
    { "mv", cmd_move },
    { "set_accel", cmd_set_accel },
    { "set_max_speed", cmd_set_max_speed },
    { "set_amp", cmd_set_amp },
    { "go_period", cmd_go_period},
    { "stop", cmd_stop },
  };
  void setup(){
    serial_commands_.SetDefaultHandler(&cmd_unrecognized);
    for (auto &cmd : cmd_func_list) {
      Serial.println(cmd.command);
      serial_commands_.AddCommand(&cmd);
    }
  }
  void check(){
    serial_commands_.ReadSerial();
  }
}


