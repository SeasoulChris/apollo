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

#ifndef INCLUDE_CYBERTRON_COMMON_MESSAGE_BASE_H_
#define INCLUDE_CYBERTRON_COMMON_MESSAGE_BASE_H_

#include <iostream>
#include <sstream>

#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/dynamic_message.h>

#include "cybertron/common/macros.h"
#include "cybertron/common/logger.h"
#include "cybertron/common/cybertron_proto.h"
// #include "cybertron/time/time.h"

#define INTER_PROCESS_CACHED_SIZE -123456789

namespace cybertron {

using namespace cybertron::proto;

class MessageBase : public google::protobuf::Message {
 public:
  SMART_PTR_DEFINITIONS(MessageBase);
  class Descriptor {
   public:
    std::string full_name() const { return "MessageBase"; }
  };
  static const Descriptor* descriptor();

  MessageBase();
  MessageBase(const std::shared_ptr<const google::protobuf::Message>& msg,
              const std::string& type_name);
  MessageBase(const MessageBase& msg)
      : google::protobuf::Message(), _msg(msg._msg), _type_name(msg._type_name), _hdr(msg._hdr) {}

  ~MessageBase();

  std::string GetTypeName() const;
  void SetTypeName(const std::string& type_name);

  MessageBase* New() const;
  int GetCachedSize() const;
  ::google::protobuf::Metadata GetMetadata() const;

  template <typename M>
  std::shared_ptr<M const> get_msg() const {
    return std::shared_ptr<M const>(std::dynamic_pointer_cast<M const>(_msg));
  }

  bool SerializeToString(std::string* output) const;
  bool ParseFromString(const std::string& msgstr);
  bool ParseFromString(const std::string& message_type,
                       const std::string& msgstr);
  static google::protobuf::Message* CreateMessage(const std::string& type_name);

  static void SerializeMessage(const google::protobuf::Message& message,
                               std::string* serialized_string);
  static void ParseMessage(const std::string& message_type,
                           const std::string& serialized_string);
  static void ParseMessage(const std::string& serialized_string,
                           google::protobuf::Message* message);

  cybertron::proto::CyberHeader* mutable_cyber_header() { return &_hdr; }
  const cybertron::proto::CyberHeader& cyber_header() const { return _hdr; }

  std::shared_ptr<const google::protobuf::Message> _msg = nullptr;
  std::string _type_name;
  cybertron::proto::CyberHeader _hdr;
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_COMMON_MESSAGE_BASE_H_
