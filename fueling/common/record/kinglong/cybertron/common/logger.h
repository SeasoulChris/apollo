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

#ifndef INCLUDE_CYBERTRON_COMMON_LOGGER_H_
#define INCLUDE_CYBERTRON_COMMON_LOGGER_H_

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include "xlog.h"
#include "cybertron/common/define.h"
#include "cybertron/common/macros.h"
// #include "cybertron/common/async_logger.h"

//hardware error
#define IPC_ERROR     11 //工控机异常
#define CAN_ERROR     12 //canbus异常
#define VEHICLE_ERROR 13 //车辆异常

//sensor error
#define LIDAR_ERROR  21 //lidar异常
#define CAMERA_ERROR 22 //camera异常
#define RADAR_ERROR  23 //radar异常
#define GNSS_ERROR   24 //gnss异常

//application error
#define DRIVER_ERROR       31
#define PERCEPTION_ERROR   32
#define LOCALIZATION_ERROR 33
#define PNC_ERROR          34
#define RECORDER_ERROR     35
#define CORAL_ERROR        36

//framework error
#define CYBERTRON_ERROR 41

#define LEFT_BRCAKET "["
#define RIGHT_BRCAKET "] "

#ifndef MODULE_NAME
#define MODULE_NAME "mainboard"
#endif

#undef LOG_DEBUG
#undef LOG_INFO
#undef LOG_WARN
#undef LOG_ERROR
#undef LOG_FATAL
#define LOG_DEBUG XLOG_MODULE(MODULE_NAME, DEBUG)
#define LOG_INFO ALOG_MODULE(MODULE_NAME, INFO)
#define LOG_WARN ALOG_MODULE(MODULE_NAME, WARN)
#define LOG_ERROR ALOG_MODULE(MODULE_NAME, ERROR)
#define LOG_FATAL ALOG_MODULE(MODULE_NAME, ERROR)

#undef LOG_DEBUG_FORMAT
#undef LOG_INFO_FORMAT
#undef LOG_WARN_FORMAT
#undef LOG_ERROR_FORMAT
#undef LOG_FATAL_FORMAT
#define LOG_DEBUG_FORMAT(args...) XLOG_MODULE_DEBUG(MODULE_NAME, ##args)
#define LOG_INFO_FORMAT(args...) ALOG_MODULE_INFO(MODULE_NAME, ##args)
#define LOG_WARN_FORMAT(args...) ALOG_MODULE_WARN(MODULE_NAME, ##args)
#define LOG_ERROR_FORMAT(args...) ALOG_MODULE_ERROR(MODULE_NAME, ##args)
#define LOG_FATAL_FORMAT(args...) ALOG_MODULE_ERROR(MODULE_NAME, ##args)

#ifndef ALOG_MODULE_STREAM
#define ALOG_MODULE_STREAM(log_severity) ALOG_MODULE_STREAM_##log_severity
#endif

#ifndef ALOG_MODULE
#define ALOG_MODULE(module, log_severity) \
  ALOG_MODULE_STREAM(log_severity)(module)
#endif

#ifndef ALOG_MODULE_INFO
#define ALOG_MODULE_INFO(module, fmt, args...) \
do { \
  ::cybertron::Logger::Inst()->WriteLog(module, google::INFO, \
                                  __FILE__, __LINE__, fmt, ##args); \
} while (0)
#endif

#ifndef ALOG_MODULE_WARN
#define ALOG_MODULE_WARN(module, fmt, args...) \
do { \
  ::cybertron::Logger::Inst()->WriteLog(module, google::WARNING, \
                                     __FILE__, __LINE__, fmt, ##args); \
} while (0)
#endif

#ifndef ALOG_MODULE_ERROR
#define ALOG_MODULE_ERROR(module, fmt, args...) \
do { \
  ::cybertron::Logger::Inst()->WriteLog(module, google::ERROR, \
                                   __FILE__, __LINE__, fmt, ##args); \
} while (0)
#endif

#define ALOG_MODULE_STREAM_INFO(module) \
  google::LogMessage(__FILE__, __LINE__, google::INFO).stream() \
  << LEFT_BRCAKET << module << RIGHT_BRCAKET

#define ALOG_MODULE_STREAM_WARN(module) \
  google::LogMessage(__FILE__, __LINE__, google::WARNING).stream() \
  << LEFT_BRCAKET << module << RIGHT_BRCAKET

#define ALOG_MODULE_STREAM_ERROR(module) \
  google::LogMessage(__FILE__, __LINE__, google::ERROR).stream() \
  << LEFT_BRCAKET << module << RIGHT_BRCAKET


#define TRACE_LOG_DEBUG(msg)                                             \
  LOG_DEBUG << "[meta_stamp:" << msg->cyber_header().meta_stamp() << "]" \
            << "[stamp:" << msg->cyber_header().stamp() << "]"           \
            << "[index:" << msg->cyber_header().index() << "]"           \
            << "[frame_id:" << msg->cyber_header().frame_id() << "] "

#define TRACE_LOG_INFO(msg)                                             \
  LOG_INFO << "[meta_stamp:" << msg->cyber_header().meta_stamp() << "]" \
           << "[stamp:" << msg->cyber_header().stamp() << "]"           \
           << "[index:" << msg->cyber_header().index() << "]"           \
           << "[frame_id:" << msg->cyber_header().frame_id() << "] "

#define TRACE_LOG_WARNING(msg)                                          \
  LOG_WARN << "[meta_stamp:" << msg->cyber_header().meta_stamp() << "]" \
           << "[stamp:" << msg->cyber_header().stamp() << "]"           \
           << "[index:" << msg->cyber_header().index() << "]"           \
           << "[frame_id:" << msg->cyber_header().frame_id() << "] "

#define TRACE_LOG_ERROR(msg)                                             \
  LOG_ERROR << "[meta_stamp:" << msg->cyber_header().meta_stamp() << "]" \
            << "[stamp:" << msg->cyber_header().stamp() << "]"           \
            << "[index:" << msg->cyber_header().index() << "]"           \
            << "[frame_id:" << msg->cyber_header().frame_id() << "] "

#if !defined(RETURN_IF_NULL)
#define RETURN_IF_NULL(ptr)          \
  if (ptr == nullptr) {              \
    LOG_WARN << #ptr << " is nullptr."; \
    return;                          \
  }
#endif
#if !defined(RETURN_VAL_IF_NULL)
#define RETURN_VAL_IF_NULL(ptr, val) \
  if (ptr == nullptr) {              \
    LOG_WARN << #ptr << " is nullptr."; \
    return val;                      \
  }
#endif
#if !defined(RETURN_IF)
#define RETURN_IF(condition)           \
  if (condition) {                     \
    LOG_WARN << #condition << " is met."; \
    return;                            \
  }
#endif
#if !defined(RETURN_VAL_IF)
#define RETURN_VAL_IF(condition, val)  \
  if (condition) {                     \
    LOG_WARN << #condition << " is met."; \
    return val;                        \
  }
#endif

#if !defined(_RETURN_VAL_IF_NULL2__)
#define _RETURN_VAL_IF_NULL2__
#define RETURN_VAL_IF_NULL2(ptr, val) \
  if (ptr == nullptr) {               \
    return (val);                     \
  }
#endif

#if !defined(_RETURN_VAL_IF2__)
#define _RETURN_VAL_IF2__
#define RETURN_VAL_IF2(condition, val) \
  if (condition) {                     \
    return (val);                      \
  }
#endif

#if !defined(_RETURN_IF2__)
#define _RETURN_IF2__
#define RETURN_IF2(condition) \
  if (condition) {            \
    return;                   \
  }
#endif

#if !defined(ERROR_AND_RETURN_IF)
#define ERROR_AND_RETURN_IF(condition, code, sub_code)                  \
  if (condition) {                                                      \
    LOG_ERROR << #code << #sub_code << " " << #condition << " is met."; \
    return;                                                             \
  }
#endif

#if !defined(ERROR_AND_RETURN_IF_NULL)
#define ERROR_AND_RETURN_IF_NULL(ptr, code, sub_code)                 \
  if (ptr == nullptr) {                                               \
    LOG_ERROR << #code << #sub_code << " " << #ptr << " is nullptr."; \
    return;                                                           \
  }
#endif

#if !defined(ERROR_AND_RETURN_VAL_IF)
#define ERROR_AND_RETURN_VAL_IF(condition, val, code, sub_code)         \
  if (condition) {                                                      \
    LOG_ERROR << #code << #sub_code << " " << #condition << " is met."; \
    return val;                                                         \
  }
#endif

#if !defined(ERROR_AND_RETURN_VAL_IF_NULL)
#define ERROR_AND_RETURN_VAL_IF_NULL(ptr, val, code, sub_code)        \
  if (ptr == nullptr) {                                               \
    LOG_ERROR << #code << #sub_code << " " << #ptr << " is nullptr."; \
    return val;                                                       \
  }
#endif

namespace cybertron {

class Node;

template<typename MessageT> class Writer;

class LogImpl {
 public:
  SMART_PTR_DEFINITIONS_NOT_COPYABLE(LogImpl);
  virtual int Initialize(const char* binary_name) = 0;
  virtual std::string name() = 0;
  virtual void Shutdown() = 0;
  virtual void set_log_destination(int32_t level, const std::string file) {}
  virtual void set_log_symlink(int32_t level, const std::string file) {}
  // virtual int EnableAsyncLog(bool send_errcode) = 0;
  virtual int EnableSignalHandle() = 0;
  virtual void WriteLog(const char* module, google::LogSeverity severity,
            const char* file, int line, const char* fmt, ...) = 0;
};

class Xlog : public LogImpl {
 public:
  SMART_PTR_DEFINITIONS(Xlog);
  int Initialize(const char* binary_name);
  std::string name();
  void Shutdown();
  void set_log_destination(int32_t level, const std::string file) {}
  void set_log_symlink(int32_t level, const std::string file) {}
  // int EnableAsyncLog(bool send_errcode);
  int EnableSignalHandle();
  void WriteLog(const char* module, google::LogSeverity severity,
        const char* file, int line, const char* fmt, ...);
 private:
  // AsyncLogger* async_logger_[google::NUM_SEVERITIES];
  std::mutex mutex_;
  bool enable_signal_handle_ = false;
};

class Logger {
 public:
  static int Create();
  static int Init(const char* binary_name);
  static LogImpl::SharedPtr Inst() {
    static LogImpl::SharedPtr inst = Xlog::SharedPtr(new Xlog);
    return inst;
  }
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_COMMON_LOGGER_H_
