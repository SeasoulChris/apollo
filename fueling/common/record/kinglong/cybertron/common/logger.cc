/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "cybertron/common/logger.h"
#include "cybertron/common/common.h"
// #include "cybertron/node/node.h"

namespace cybertron {

void SignalHandle(const char* data, int size) {
  ALOG_MODULE("core_stack", ERROR) << data;
  LOG_ERROR << data;
}

int Logger::Create() { return SUCC; }

int Logger::Init(const char* binary_name) {
  return Inst()->Initialize(binary_name);
}

int Xlog::Initialize(const char* binary_name) {
  if (adu::xlog::init_xlogging(binary_name) < 0) {
    return FAIL;
  }
  return SUCC;
}

std::string Xlog::name() { return "XLOG"; }

// int Xlog::EnableAsyncLog(bool send_errcode) {
//   uint64_t buf_size = 2 * 1024 * 1024;
//   for (auto severity : {google::INFO /*, google::WARNING, google::ERROR*/}) {
//     async_logger_[severity] =
//         new AsyncLogger(google::base::GetLogger(severity), buf_size, send_errcode);
//     async_logger_[severity]->Start();
//     google::base::SetLogger(severity, async_logger_[severity]);
//   }
//   LOG_INFO << "enable async log successfully.";
//   return SUCC;
// }

int Xlog::EnableSignalHandle() {
  if (enable_signal_handle_) {
    return SUCC;
  }
  google::InstallFailureSignalHandler();
  google::InstallFailureWriter(&::cybertron::SignalHandle);
  enable_signal_handle_ = true;
  return SUCC;
}

void Xlog::WriteLog(const char* module, google::LogSeverity severity,
                    const char* file, int line, const char* fmt, ...) {
  char fmt_buffer[9000];
  va_list arg_ptr;
  va_start(arg_ptr, fmt);
  vsnprintf(fmt_buffer, sizeof(fmt_buffer), fmt, arg_ptr);
  va_end(arg_ptr);
  google::LogMessage(file, line, severity).stream()
      << LEFT_BRCAKET << module << RIGHT_BRCAKET << fmt_buffer;
}

void Xlog::Shutdown() { adu::xlog::shutdown_xlogging(); }

}  // namespace cybertron
