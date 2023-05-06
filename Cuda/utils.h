#pragma once
#include <glog/logging.h>

#include <iomanip>
#include <mutex>
std::once_flag glog_init_flag;
void InitGLOG(const std::string& prog_name) {
  std::call_once(glog_init_flag, [&]() {
    google::InitGoogleLogging(strdup(prog_name.c_str()));
  });
}
