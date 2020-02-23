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

#pragma once

#include <iostream>
#include <sstream>

#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/dynamic_message.h>

#include "fueling/common/record/kinglong/cybertron/common/macros.h"
// #include "fueling/common/record/kinglong/cybertron/common/logger.h"
#include "fueling/common/record/kinglong/cybertron/common/message_base.h"

namespace {
std::shared_ptr<const std::string> PY_MESSAGE_FULLNAME(
    new std::string("PyMessageBase"));
}

namespace cybertron {

class PyMessageShifter {
 public:
  class Descriptor {
   public:
    std::string full_name() const { return *PY_MESSAGE_FULLNAME.get(); }
    std::string name() const { return *PY_MESSAGE_FULLNAME.get(); }
  };
  static const Descriptor* descriptor() {
    static Descriptor desc;
    return &desc;
  }

  PyMessageShifter() : _type_name("") {}
  PyMessageShifter(const std::string& msg, const std::string& type_name)
      : _type_name(type_name), _data(msg) {
    PY_MESSAGE_FULLNAME.reset(new std::string(_type_name));
  }
  PyMessageShifter(const PyMessageShifter& msg)
      : _type_name(msg._type_name), _data(msg._data), _hdr(msg._hdr) {}

  ~PyMessageShifter() {}

  std::string GetTypeName();
  void SetTypeName(const std::string& type_name);

  bool SerializeToString(std::string* output) const {
    if (!output) {
      return false;
    }
    *output = _data;
    return true;
  }

  bool ParseFromString(const std::string& msgstr) {
    _data = msgstr;
    return true;
  }

  std::string data() const { return _data; }

  void set_data(const std::string& msg) { _data = msg; }

  fueling::common::record::kinglong::proto::cybertron::CyberHeader* mutable_cyber_header() { return &_hdr; }
  const fueling::common::record::kinglong::proto::cybertron::CyberHeader& cyber_header() const { return _hdr; }

  std::string _type_name;
  std::string _data;
  fueling::common::record::kinglong::proto::cybertron::CyberHeader _hdr;
};

}  // namespace cybertron
