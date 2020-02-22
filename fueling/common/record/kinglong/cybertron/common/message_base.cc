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

#include "cybertron/common/message_base.h"
#include "cybertron/common/protobuf_factory.h"
// #include "cybertron/time/time.h"

namespace cybertron {

MessageBase::MessageBase() : _type_name("MessageBase") {
  //_hdr.set_meta_stamp(Time::Now().ToNanosecond());
  //_hdr.set_stamp(Time::Now().ToNanosecond());
}

MessageBase::MessageBase(
    const std::shared_ptr<const google::protobuf::Message>& msg,
    const std::string& type_name)
    : _msg(msg), _type_name(type_name) {
  //_hdr.set_meta_stamp(Time::Now().ToNanosecond());
  //_hdr.set_stamp(Time::Now().ToNanosecond());
}

MessageBase::~MessageBase() {}

const MessageBase::Descriptor* MessageBase::descriptor() {
  static Descriptor desc;
  return &desc;
}

std::string MessageBase::GetTypeName() const { 
  return _type_name; 
}

void MessageBase::SetTypeName(const std::string& type_name) {
  _type_name = type_name;
}

MessageBase* MessageBase::New() const { return new MessageBase(); }

int MessageBase::GetCachedSize() const { return INTER_PROCESS_CACHED_SIZE; }

::google::protobuf::Metadata MessageBase::GetMetadata() const {
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = nullptr;
  metadata.reflection = nullptr;
  return metadata;
}

bool MessageBase::SerializeToString(std::string* output) const {
  std::string hdr_str;
  RETURN_VAL_IF2(!_hdr.SerializeToString(&hdr_str), false);

  std::string msg_str;
  RETURN_VAL_IF2(!_msg->SerializeToString(&msg_str), false);

  std::ostringstream os;
  uint64_t len = hdr_str.size();
  os.write((char*)&len, sizeof(uint64_t));
  os.write((char*)hdr_str.data(), len);
  len = _type_name.size();
  os.write((char*)&len, sizeof(uint64_t));
  os.write((char*)_type_name.c_str(), len);
  len = msg_str.size();
  os.write((char*)&len, sizeof(uint64_t));
  os.write((char*)msg_str.data(), len);
  output->resize(os.str().size());
  *output = os.str();
  return true;
}

bool MessageBase::ParseFromString(const std::string& msgstr) {
  uint64_t len = 0;
  std::istringstream is;
  is.read((char*)&len, sizeof(uint64_t));
  std::string hdr_str("", len);
  is.read(&hdr_str[0], len);
  RETURN_VAL_IF2(!_hdr.ParseFromString(hdr_str), false);

  is.read((char*)&len, sizeof(uint64_t));
  is.read(&_type_name[0], len);
  is.read((char*)&len, sizeof(uint64_t));
  std::string msg_str("", len);
  is.read(&msg_str[0], len);
  RETURN_VAL_IF2(!this->ParseFromString(_type_name, msg_str), false);
  return false;
}

bool MessageBase::ParseFromString(const std::string& message_type,
                                  const std::string& msgstr) {
  google::protobuf::Message* msg = CreateMessage(message_type);
  if (msg == nullptr) {
    return false;
  }
  RETURN_VAL_IF2(!msg->ParseFromString(msgstr), false);

  _msg.reset(static_cast<const google::protobuf::Message*>(msg));
  return true;
}

google::protobuf::Message* MessageBase::CreateMessage(
    const std::string& type_name) {
  return ProtobufFactory::Instance()->GenerateMessageByType(type_name);
}

void MessageBase::SerializeMessage(const google::protobuf::Message& message,
                                   std::string* serialized_string) {
  const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
  const google::protobuf::Reflection* reflection = message.GetReflection();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* field = descriptor->field(i);
    bool has_field = reflection->HasField(message, field);
    if (has_field) {
      // arrays not supported
      // assert(!field->is_repeated());
      switch (field->cpp_type()) {
#define CASE_FIELD_TYPE(cpptype, method, valuetype)                            \
  case google::protobuf::FieldDescriptor::CPPTYPE_##cpptype: {                 \
    valuetype value = reflection->Get##method(message, field);                 \
    int wsize = field->name().size();                                          \
    serialized_string->append(reinterpret_cast<char*>(&wsize), sizeof(wsize)); \
    serialized_string->append(field->name().c_str(), field->name().size());    \
    wsize = sizeof(value);                                                     \
    serialized_string->append(reinterpret_cast<char*>(&wsize), sizeof(wsize)); \
    serialized_string->append(reinterpret_cast<char*>(&value), sizeof(value)); \
    break;                                                                     \
  }
        CASE_FIELD_TYPE(INT32, Int32, int);
        CASE_FIELD_TYPE(UINT32, UInt32, uint32_t);
        CASE_FIELD_TYPE(FLOAT, Float, float);
        CASE_FIELD_TYPE(DOUBLE, Double, double);
        CASE_FIELD_TYPE(BOOL, Bool, bool);
        CASE_FIELD_TYPE(INT64, Int64, int64_t);
        CASE_FIELD_TYPE(UINT64, UInt64, uint64_t);
#undef CASE_FIELD_TYPE
        case google::protobuf::FieldDescriptor::CPPTYPE_ENUM: {
          int value = reflection->GetEnum(message, field)->number();
          int wsize = field->name().size();
          serialized_string->append(reinterpret_cast<char*>(&wsize),
                                    sizeof(wsize));
          serialized_string->append(field->name().c_str(),
                                    field->name().size());
          wsize = sizeof(value);
          serialized_string->append(reinterpret_cast<char*>(&wsize),
                                    sizeof(wsize));
          serialized_string->append(reinterpret_cast<char*>(&value),
                                    sizeof(value));
          break;
        }
        case google::protobuf::FieldDescriptor::CPPTYPE_STRING: {
          std::string value = reflection->GetString(message, field);
          int wsize = field->name().size();
          serialized_string->append(reinterpret_cast<char*>(&wsize),
                                    sizeof(wsize));
          serialized_string->append(field->name().c_str(),
                                    field->name().size());
          wsize = value.size();
          serialized_string->append(reinterpret_cast<char*>(&wsize),
                                    sizeof(wsize));
          serialized_string->append(value.c_str(), value.size());
          break;
        }
        case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE: {
          std::string value;
          int wsize = field->name().size();
          serialized_string->append(reinterpret_cast<char*>(&wsize),
                                    sizeof(wsize));
          serialized_string->append(field->name().c_str(),
                                    field->name().size());
          const google::protobuf::Message& submessage =
              reflection->GetMessage(message, field);
          SerializeMessage(submessage, &value);
          wsize = value.size();
          serialized_string->append(reinterpret_cast<char*>(&wsize),
                                    sizeof(wsize));
          serialized_string->append(value.c_str(), value.size());
          break;
        }
      }
    }
  }
}

void MessageBase::ParseMessage(const std::string& message_type,
                               const std::string& serialized_string) {
  google::protobuf::Message* message =
      ProtobufFactory::Instance()->GenerateMessageByType(message_type);
  RETURN_IF2(message == nullptr);

  ParseMessage(serialized_string, message);
  
  if (message) {
    delete message;
  }
}

void MessageBase::ParseMessage(const std::string& serialized_string,
                               google::protobuf::Message* message) {
  const google::protobuf::Descriptor* descriptor = message->GetDescriptor();
  const google::protobuf::Reflection* reflection = message->GetReflection();
  std::map<std::string, const google::protobuf::FieldDescriptor*> field_map;
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* field = descriptor->field(i);
    field_map[field->name()] = field;
  }
  const google::protobuf::FieldDescriptor* field = NULL;
  size_t pos = 0;
  while (pos < serialized_string.size()) {
    int name_size = *(reinterpret_cast<const int*>(
        serialized_string.substr(pos, sizeof(int)).c_str()));
    pos += sizeof(int);
    std::string name = serialized_string.substr(pos, name_size);
    pos += name_size;
    int value_size = *(reinterpret_cast<const int*>(
        serialized_string.substr(pos, sizeof(int)).c_str()));
    pos += sizeof(int);
    std::string value = serialized_string.substr(pos, value_size);
    pos += value_size;
    std::map<std::string, const google::protobuf::FieldDescriptor*>::iterator
        iter = field_map.find(name);
    if (iter == field_map.end()) {
      LOG_DEBUG << "no field found. stderr: " << stderr;
      continue;
    } else {
      field = iter->second;
    }
    // assert(!field->is_repeated());
    switch (field->cpp_type()) {
#define CASE_FIELD_TYPE(cpptype, method, valuetype)                                    \
  case google::protobuf::FieldDescriptor::CPPTYPE_##cpptype: {                         \
    reflection->Set##method(                                                           \
        message, field, *(reinterpret_cast<const valuetype*>(value.c_str())));         \
    std::cout << field->name() << ": "                                                 \
              << std::to_string(*(reinterpret_cast<const valuetype*>(value.c_str())))  \
              << std::endl;                                                            \
    break;                                                                             \
  }
      CASE_FIELD_TYPE(INT32, Int32, int);
      CASE_FIELD_TYPE(UINT32, UInt32, uint32_t);
      CASE_FIELD_TYPE(BOOL, Bool, bool);
      CASE_FIELD_TYPE(INT64, Int64, int64_t);
      CASE_FIELD_TYPE(UINT64, UInt64, uint64_t);
#undef CASE_FIELD_TYPE
      case google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE: {
        reflection->SetDouble(message, field, std::stod(value));
        std::cout << field->name() << ": " << std::stod(value) << std::endl;
        break;
      }
      case google::protobuf::FieldDescriptor::CPPTYPE_FLOAT: {
        reflection->SetFloat(message, field, std::stof(value));
        std::cout << field->name() << ": " << std::stof(value) << std::endl;
        break;
      }
      case google::protobuf::FieldDescriptor::CPPTYPE_ENUM: {
        const google::protobuf::EnumValueDescriptor* enum_value_descriptor =
            field->enum_type()->FindValueByNumber(
                *(reinterpret_cast<const int*>(value.c_str())));
        reflection->SetEnum(message, field, enum_value_descriptor);
        std::cout << field->name() << ": "
                  << *(reinterpret_cast<const int*>(value.c_str()))
                  << std::endl;
        break;
      }
      case google::protobuf::FieldDescriptor::CPPTYPE_STRING: {
        reflection->SetString(message, field, value);
        std::cout << field->name() << ": " << value << std::endl;
        break;
      }
      case google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE: {
        google::protobuf::Message* submessage =
            reflection->MutableMessage(message, field);
        ParseMessage(value, submessage);
        break;
      }
      default: { break; }
    }
  }
}

}  // namespace cybertron
