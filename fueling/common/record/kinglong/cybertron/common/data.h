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

#ifndef INCLUDE_CYBERTRON_COMMON_DATA_H_
#define INCLUDE_CYBERTRON_COMMON_DATA_H_

#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <mutex>
#include <unordered_map>

#include "fueling/common/record/kinglong/cybertron/common/macros.h"
#include "fueling/common/record/kinglong/cybertron/common/error_code.h"
#include "fueling/common/record/kinglong/cybertron/common/define.h"

namespace cybertron {

class GlobalData {
 public:
  SMART_PTR_DEFINITIONS(GlobalData)
  ~GlobalData();

  typedef std::unordered_map<std::string, std::string> DataMap;
  typedef std::unordered_map<std::string, std::string> ChannelTypeMap;

  void set_process_type(const std::string& process_type);
  std::string get_process_type();

  void set_process_id(const std::string& process_id);
  std::string get_process_id();

  void set_process_name(const std::string& process_name);
  std::string get_process_name();

  void set_host_ip(const std::string& host_ip);
  std::string get_host_ip();

  void set_host_name(const std::string& host_name);
  std::string get_host_name();

  int set_channel_type(const std::string& channelname,
                       const std::string& msgtype);
  std::string get_channel_type(const std::string& channelname);

  void clear();

 private:
  std::mutex _mutex;
  DataMap _data_map;
  std::string _process_name;
  ChannelTypeMap _channel_type_name;

  DECLARE_SINGLETON(GlobalData)
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_COMMON_DATA_H_
