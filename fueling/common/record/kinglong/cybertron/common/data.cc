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

#include "fueling/common/record/kinglong/cybertron/common/data.h"

namespace cybertron {

GlobalData::GlobalData() {
  int buffsize = 1024;
  char host_name[buffsize];
  gethostname(host_name, buffsize);
  struct hostent* host = gethostbyname(host_name);
  if (host != NULL) {
    std::string host_ip = inet_ntoa(*((struct in_addr*)host->h_addr_list[0]));
    _data_map["host_ip"] = host_ip;
  }
  _data_map["host_name"] = host_name;
  _data_map["process_type"] = "mainboard";
}

GlobalData::~GlobalData() {}

void GlobalData::set_process_type(const std::string& process_type) {
  usleep(1000);
  _data_map["process_type"] = process_type;
}
std::string GlobalData::get_process_type() {
  auto itr = _data_map.find("process_type");
  if (itr == _data_map.end()) {
    return "tools";
  }
  return itr->second;
}

void GlobalData::set_process_id(const std::string& process_id) {
  usleep(1000);
  _data_map["process_id"] = process_id;
}
std::string GlobalData::get_process_id() {
  auto itr = _data_map.find("process_id");
  if (itr == _data_map.end()) {
    return "default_" + std::to_string(getpid());
  }
  return itr->second;
}

void GlobalData::set_process_name(const std::string& process_name) {
  _data_map["process_name"] = process_name;
}
std::string GlobalData::get_process_name() {
  auto itr = _data_map.find("process_name");
  if (itr == _data_map.end()) {
    return "default_" + std::to_string(getpid());
  }
  return itr->second;
}

void GlobalData::set_host_ip(const std::string& host_ip) {
  _data_map["host_ip"] = host_ip;
}
std::string GlobalData::get_host_ip() { return _data_map["host_ip"]; }

void GlobalData::set_host_name(const std::string& host_name) {
  _data_map["host_name"] = host_name;
}
std::string GlobalData::get_host_name() { return _data_map["host_name"]; }

int GlobalData::set_channel_type(const std::string& channelname,
                                 const std::string& msgtype) {
  auto itr = _channel_type_name.find(channelname);
  if (itr != _channel_type_name.end()) {
    if (itr->second != msgtype) {
      return FAIL;
    }
  }
  _channel_type_name[channelname] = msgtype;
  return SUCC;
}

std::string GlobalData::get_channel_type(const std::string& channelname) {
  auto itr = _channel_type_name.find(channelname);
  if (itr == _channel_type_name.end()) {
    return "";
  }

  return itr->second;
}

void GlobalData::clear() {
  auto itr = _data_map.find("process_id");
  if (itr != _data_map.end()) {
    _data_map.erase(itr);
  }
}

}  // namespace cybertron
