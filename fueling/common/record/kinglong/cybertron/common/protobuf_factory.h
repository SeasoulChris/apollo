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

#include <string>
#include <mutex>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/io/tokenizer.h>
#include <google/protobuf/compiler/parser.h>
// #include "cybertron/common/logger.h"
#include "fueling/common/record/kinglong/cybertron/common/macros.h"
#include "fueling/common/record/kinglong/cybertron/common/error_code.h"
#include "fueling/common/record/kinglong/proto/cybertron/proto_desc.pb.h"

namespace cybertron {

class ErrorCollector : public google::protobuf::DescriptorPool::ErrorCollector {
  using ErrorLocation =
      google::protobuf::DescriptorPool::ErrorCollector::ErrorLocation;
  void AddError(const std::string& filename, const std::string& element_name,
                const google::protobuf::Message* descriptor,
                ErrorLocation location, const std::string& message) override;

  void AddWarning(const std::string& filename, const std::string& element_name,
                  const google::protobuf::Message* descriptor,
                  ErrorLocation location, const std::string& message) override;
};

class ProtobufFactory {
 public:
  SMART_PTR_DEFINITIONS(ProtobufFactory)
  using ProtoDesc = fueling::common::record::kinglong::proto::cybertron::ProtoDesc;

  ~ProtobufFactory();

  // Recursively register FileDescriptorProto and all its dependencies to
  // factory.
  int RegisterMessage(const std::string& proto_desc_str);
  int RegisterPythonMessage(const std::string& proto_str);

  // Convert the serialized FileDescriptorProto to real descriptors and place
  // them in factory.
  // It is an error if a FileDescriptorProto contains references to types or
  // other files that
  // are not found in the Factory.
  int RegisterMessage(const google::protobuf::Message& message);
  int RegisterMessage(const google::protobuf::Descriptor& desc);
  int RegisterMessage(
      const google::protobuf::FileDescriptorProto& file_desc_proto);

  // Serialize all descriptors of the given message to string.
  static void GetDescriptorString(const google::protobuf::Message& message,
                                  std::string* desc_str);

  // Serialize all descriptors of the descriptor to string.
  static void GetDescriptorString(const google::protobuf::Descriptor* desc,
                                  std::string* desc_str);

  // Get Serialized descriptors of messages with the given type.
  void GetDescriptorString(const std::string& type, std::string* desc_str);

  // Given a type name, constructs the default (prototype) Message of that type.
  // Returns nullptr if no such message exists.
  google::protobuf::Message* GenerateMessageByType(
      const std::string& type) const;

  // Find a top-level message type by name. Returns NULL if not found.
  const google::protobuf::Descriptor* FindMessageTypeByName(
      const std::string& type) const;

  // Find a service definition by name. Returns NULL if not found.
  const google::protobuf::ServiceDescriptor* FindServieByName(
      const std::string& name) const;

  void GetPythonDesc(const std::string& type, std::string* desc_str);

 private:
  int RegisterMessage(const ProtoDesc& proto_desc);
  google::protobuf::Message* GetMessageByGeneratedType(
      const std::string& type) const;
  static int GetProtoDesc(const google::protobuf::FileDescriptor* file_desc,
                          ProtoDesc* proto_desc);

  std::mutex register_mutex_;
  std::unique_ptr<google::protobuf::DescriptorPool> _pool = nullptr;
  std::unique_ptr<google::protobuf::DynamicMessageFactory> _factory = nullptr;

  DECLARE_SINGLETON(ProtobufFactory);
};

}  // namespace cybertron
