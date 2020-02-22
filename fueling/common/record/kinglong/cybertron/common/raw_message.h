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

#ifndef INCLUDE_CYBERTRON_COMMON_RAW_MESSAGE_H_
#define INCLUDE_CYBERTRON_COMMON_RAW_MESSAGE_H_

#include "cybertron/common/message_base.h"

namespace cybertron {

class RawMessage : public MessageBase {
 public:
  SMART_PTR_DEFINITIONS(RawMessage);
  class Descriptor {
   public:
    std::string full_name() const { return "RawMessage"; }
  };
  static const Descriptor* descriptor() {
    static Descriptor desc;
    return &desc;
  }

  void set_msg(const std::string& msg) { _msg = msg; }
  std::string get_msg() const { return _msg; }
  void set_type_name(const std::string& type_name) { _type_name = type_name; }
  std::string get_type_name() const { return _type_name; }

 private:
  std::string _msg = "";
  std::string _type_name = "";
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_COMMON_RAW_MESSAGE_H_
