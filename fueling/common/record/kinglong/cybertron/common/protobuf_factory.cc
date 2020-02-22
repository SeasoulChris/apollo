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

#include "cybertron/common/define.h"
#include "cybertron/common/protobuf_factory.h"

namespace cybertron {

ProtobufFactory::ProtobufFactory() {
  _pool.reset(new google::protobuf::DescriptorPool());
  _factory.reset(new google::protobuf::DynamicMessageFactory(_pool.get()));
}

ProtobufFactory::~ProtobufFactory() {
  _factory.reset();
  _pool.reset();
}

int ProtobufFactory::RegisterMessage(const google::protobuf::Message& message) {
  const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
  return RegisterMessage(*descriptor);
}

int ProtobufFactory::RegisterMessage(const google::protobuf::Descriptor& desc) {
  google::protobuf::FileDescriptorProto file_desc_proto;
  desc.file()->CopyTo(&file_desc_proto);
  return RegisterMessage(file_desc_proto);
}

int ProtobufFactory::RegisterMessage(const ProtoDesc& proto_desc) {
  for (int i = 0; i < proto_desc.dependencies_size(); ++i) {
    auto dep = proto_desc.dependencies(i);
    if (RegisterMessage(dep) != SUCC) {
      return FAIL;
    }
  }

  google::protobuf::FileDescriptorProto file_desc_proto;
  file_desc_proto.ParseFromString(proto_desc.desc());
  return RegisterMessage(file_desc_proto);
}

int ProtobufFactory::RegisterPythonMessage(const std::string& proto_str) {
  google::protobuf::FileDescriptorProto file_desc_proto;
  file_desc_proto.ParseFromString(proto_str);
  return RegisterMessage(file_desc_proto);
}

int ProtobufFactory::RegisterMessage(const std::string& proto_desc_str) {
  ProtoDesc proto_desc;
  proto_desc.ParseFromString(proto_desc_str);
  return RegisterMessage(proto_desc);
}

// Internal method
int ProtobufFactory::RegisterMessage(
    const google::protobuf::FileDescriptorProto& file_desc_proto) {
  std::lock_guard<std::mutex> lg(register_mutex_);
  ErrorCollector ec;
  auto file_desc = _pool->BuildFileCollectingErrors(file_desc_proto, &ec);
  if (!file_desc) {
    /*
    LOG_ERROR << CYBERTRON_ERROR << PROTOBUF_REGISTER_MSG_ERROR 
      << "Failed to register protobuf messages [" << file_desc_proto.name() << "]";
    */
    return FAIL;
  }

  return SUCC;
}

// Internal method
int ProtobufFactory::GetProtoDesc(
    const google::protobuf::FileDescriptor* file_desc, ProtoDesc* proto_desc) {
  google::protobuf::FileDescriptorProto file_desc_proto;
  file_desc->CopyTo(&file_desc_proto);
  std::string str("");
  RETURN_VAL_IF2(!file_desc_proto.SerializeToString(&str), FAIL);

  proto_desc->set_desc(str);

  for (int i = 0; i < file_desc->dependency_count(); ++i) {
    auto desc = proto_desc->add_dependencies();
    RETURN_VAL_IF2(GetProtoDesc(file_desc->dependency(i), desc) != SUCC, FAIL);
  }

  return SUCC;
}

void ProtobufFactory::GetDescriptorString(
    const google::protobuf::Descriptor* desc, std::string* desc_str) {
  cybertron::proto::ProtoDesc proto_desc;
  if (GetProtoDesc(desc->file(), &proto_desc) != SUCC) {
    LOG_ERROR << CYBERTRON_ERROR << PROTOBUF_GET_DESC_ERROR 
      << "Failed to get descriptor from message";
    return;
  }

  if (!proto_desc.SerializeToString(desc_str)) {
    LOG_ERROR << CYBERTRON_ERROR << PROTOBUF_GET_DESC_ERROR 
      << "Failed to get descriptor from message";
  }
}

void ProtobufFactory::GetDescriptorString(
    const google::protobuf::Message& message, std::string* desc_str) {
  const google::protobuf::Descriptor* desc = message.GetDescriptor();
  return GetDescriptorString(desc, desc_str);
}

void ProtobufFactory::GetPythonDesc(const std::string& type, std::string* desc_str) {
  auto desc = _pool->FindMessageTypeByName(type);
  RETURN_IF2(desc == NULL);
  google::protobuf::DescriptorProto dp;
  desc->CopyTo(&dp);
  if (!dp.SerializeToString(desc_str)) {
    LOG_WARN << CYBERTRON_ERROR << PROTOBUF_GET_DESC_ERROR 
      << "Failed to get descriptor from message";
  }
}

void ProtobufFactory::GetDescriptorString(const std::string& type,
                                          std::string* desc_str) {
  auto desc =
      google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(
          type);
  if (desc != NULL) {
    return GetDescriptorString(desc, desc_str);
  }

  desc = _pool->FindMessageTypeByName(type);
  if (desc == NULL) {
    return;
  }
  return GetDescriptorString(desc, desc_str);
}

// Internam method
google::protobuf::Message* ProtobufFactory::GenerateMessageByType(
    const std::string& type) const {
  google::protobuf::Message* message = GetMessageByGeneratedType(type);
  if (message != nullptr) {
    return message;
  }

  const google::protobuf::Descriptor* descriptor =
      _pool->FindMessageTypeByName(type);
  if (descriptor == NULL) {
    LOG_ERROR << CYBERTRON_ERROR << PROTOBUF_GENERATE_MSG_ERROR << " cannot find [" 
      << type << "] descriptor";
    return nullptr;
  }

  const google::protobuf::Message* prototype =
      _factory->GetPrototype(descriptor);
  if (prototype == NULL) {
    LOG_ERROR << CYBERTRON_ERROR << PROTOBUF_GENERATE_MSG_ERROR << " cannot find ["
      << type << "] prototype";
    return nullptr;
  }

  return prototype->New();
}

google::protobuf::Message* ProtobufFactory::GetMessageByGeneratedType(
    const std::string& type) const {
  auto descriptor =
      google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(
          type);
  if (descriptor == NULL) {
    // LOG_WARN << "cannot find [" << type << "] descriptor";
    return nullptr;
  }

  auto prototype =
      google::protobuf::MessageFactory::generated_factory()->GetPrototype(
          descriptor);
  // LOG_ERROR << CYBERTRON_ERROR <<  << "cannot find [" << type << "] prototype";
  RETURN_VAL_IF2(prototype == NULL, nullptr);

  return prototype->New();
}

const google::protobuf::Descriptor* ProtobufFactory::FindMessageTypeByName(
    const std::string& name) const {
  return _pool->FindMessageTypeByName(name);
}

const google::protobuf::ServiceDescriptor* ProtobufFactory::FindServieByName(
    const std::string& name) const {
  return _pool->FindServiceByName(name);
}

void ErrorCollector::AddError(const std::string& filename,
                              const std::string& element_name,
                              const google::protobuf::Message* descriptor,
                              ErrorLocation location,
                              const std::string& message) {
  LOG_INFO << "[" << filename << "]. Info: " << message;
}

void ErrorCollector::AddWarning(const std::string& filename,
                                const std::string& element_name,
                                const google::protobuf::Message* descriptor,
                                ErrorLocation location,
                                const std::string& message) {
  LOG_INFO << "[" << filename << "]." << "Info: " << message;
}

}  // namespace cybertron
