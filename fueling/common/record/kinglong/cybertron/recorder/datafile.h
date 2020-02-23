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

#include <unordered_map>
#include <atomic>
#include <thread>

#include "fueling/common/record/kinglong/cybertron/common/raw_message.h"
#include "fueling/common/record/kinglong/cybertron/common/protobuf_factory.h"
#include "fueling/common/record/kinglong/cybertron/common/conf_manager.h"
#include "fueling/common/record/kinglong/cybertron/recorder/chunk.h"
#include "fueling/common/record/kinglong/cybertron/recorder/param.h"
#include "fueling/common/record/kinglong/cybertron/recorder/fileopt.h"
#include "fueling/common/record/kinglong/proto/cybertron/record.pb.h"
#include "fueling/common/record/kinglong/proto/cybertron/parameter.pb.h"

namespace cybertron {

enum FileMode : uint32_t { Write = 1, Read = 2 };

class Index {
 public:
  bool operator<(const Index& idx) { return time < idx.time; }

 public:
  int time;
  int chunk_idx;
  int message_idx;
};

class DataFile {
 public:
  SMART_PTR_DEFINITIONS(DataFile)
  DataFile();
  DataFile(const RecorderParam& param);
  DataFile(const std::string& filename, const uint32_t& mode = FileMode::Read, bool if_dump_parameter_snapshot = false);
  ~DataFile();

  int open(const std::string& filename, const uint32_t& mode = FileMode::Read, bool if_dump_parameter_snapshot = false);
  void close();

  std::string GetSnapshot();
  int SetSnapshot(const std::string& snapshot);
  // write api
  template <typename T>
  int Write(const std::string& channel, const std::shared_ptr<const T>& message,
            const uint64_t time = 0);

  template <typename T>
  int Write(const std::string& channel, const T& message,
            const uint64_t time = 0);

  int Write(const std::string& channel, const std::string& message,
            const std::string& type, const uint64_t time = 0);

  void set_chunk_limit(const uint64_t& limit);
  void set_compress_type(const fueling::common::record::kinglong::proto::cybertron::CompressType& type);
  void set_chunk_interval(const uint64_t& interval);
  void set_segment_interval(const uint64_t& interval);

  // read api
  std::vector<std::string> get_channels() const;
  std::string get_channel_type(const std::string& channel) const;
  std::string GetDescByName(const std::string& name) const;
  int get_msg_num(const std::string& channel) const;
  int get_msg_num() const;
  uint64_t get_begin_time() const;
  uint64_t get_end_time() const;
  uint64_t get_file_size() const;
  bool IsActive() const;
  std::string get_file_name() const;
  std::string get_version() const { return version_; }
  fueling::common::record::kinglong::proto::cybertron::CompressType get_compress_type() const {
    return compress_type_;
  }

  // deprecated api
  void AddChannel(const std::string& channel, const std::string& type,
                  const std::string& proto_desc);

 private:
  friend class Record;
  friend class ReIndex;
  friend class Split;
  friend class DataIterator;

  int flush();
  int ReadSnapshot();
  int WriteSnapshot(bool if_dump_parameter_snapshot = false);
  int get_chunk_num() const;
  int get_msg_num(const int& chunk_index) const;
  int Write(const fueling::common::record::kinglong::proto::cybertron::SingleMsg& singlemsg);
  fueling::common::record::kinglong::proto::cybertron::SingleMsg ReadMessage(const int& chunk_index,
                                          const int& msg_index);

  void ClearAll();
  void StopWrite();
  void SplitOutfile(bool if_dump_parameter_snapshot = false);
  int ReadChunk(fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunk,
                fueling::common::record::kinglong::proto::cybertron::ChunkHeader* header);
  void ReadChunk(const int& index, fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunk,
                 fueling::common::record::kinglong::proto::cybertron::ChunkHeader* header, bool reset = true);
  void set_header(const fueling::common::record::kinglong::proto::cybertron::HeaderSection& header);
  void ShowProgress();
  void FlushChunk();

 private:
  std::string path_ = "";
  std::string version_ = "1.0.0";
  uint64_t chunk_limit_ = 50 * 1024 * 1024;
  uint64_t chunk_interval_ = 20e9L;
  uint64_t segment_interval_ = 0;
  fueling::common::record::kinglong::proto::cybertron::CompressType compress_type_ =
      fueling::common::record::kinglong::proto::cybertron::COMPRESS_NONE;

  int file_index_ = 0;
  bool is_writing_ = false;
  bool first_flush_ = true;
  bool file_is_open_ = true;
  std::string file_name_ = "";
  bool parameter_shot_open_ = true;

  // status info
  uint64_t channel_size_ = 0;
  fueling::common::record::kinglong::proto::cybertron::HeaderSection header_;
  std::unordered_map<std::string, int> msg_count_;
  std::unordered_map<std::string, std::string> types_;

  // TODO: use auto pointer instead
  fueling::common::record::kinglong::proto::cybertron::ParamSnapshot* snapshot_ = nullptr;
  // ParameterRecorderHelper::SharedPtr param_helper_ = nullptr;
  std::vector<fueling::common::record::kinglong::proto::cybertron::ParamEvent> param_events_;

  std::recursive_mutex chunk_mutex_;
  Chunk::UniquePtr chunk_ = nullptr;
  Chunk::UniquePtr chunk_backup_ = nullptr;
  OutFileOpt::UniquePtr outfileopt_ = nullptr;
  OutFileOpt::UniquePtr outfileopt_backup_ = nullptr;
  InFileOpt::UniquePtr infileopt_ = nullptr;

  std::shared_ptr<std::thread> flush_thread_ = nullptr;
  std::mutex flush_mutex_;
  std::condition_variable flush_cond_;
  std::atomic<bool> is_flushing_;
  bool run_in_sim_ = false;
};

template <typename MessageT>
int DataFile::Write(const std::string& channel,
                    const std::shared_ptr<const MessageT>& message,
                    const uint64_t time) {
  //LOG_DEBUG << "Received message from channel [" << channel << "]";
  std::string msgtype = MessageT::descriptor()->full_name();
  auto it = types_.find(channel);
  if (it != types_.end()) {
    if (it->second != msgtype) {
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_INVALID_MSG_TYPE_ERROR
      //   << " Message type [" << msgtype << "] is invalid: expect [" << it->second << "]";
      return FAIL;
    }
  } else {
    std::string proto_desc = "";
    ProtobufFactory::Instance()->GetDescriptorString(msgtype, &proto_desc);
    //if (proto_desc == "") {
      //LOG_ERROR << CYBERTRON_ERROR <<  << " message [" << msgtype << "] proto_desc empty.";
      //return FAIL;
    //}
    AddChannel(channel, msgtype, proto_desc);
  }

  std::string msgstr("");
  if (!message->SerializeToString(&msgstr)) {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR <<
    //   " Failed to serialize message for channel [" << channel << "]";
    return FAIL;
  }

  fueling::common::record::kinglong::proto::cybertron::SingleMsg singlemsg;
  singlemsg.set_channelname(channel);
  singlemsg.set_msg(msgstr);
  if (time > 0) {
    singlemsg.set_time(time);
  } else {
    // singlemsg.set_time(Time::Now().ToNanosecond());
  }

  return Write(singlemsg);
}

template <typename MessageT>
int DataFile::Write(const std::string& channel, const MessageT& message,
                    const uint64_t time) {
  //LOG_DEBUG << "Received message from channel [" << channel << "]";
  std::string msgtype = MessageT::descriptor()->full_name();
  auto it = types_.find(channel);
  if (it != types_.end()) {
    if (it->second != msgtype) {
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_INVALID_MSG_TYPE_ERROR << " Message type [" << msgtype << "] is invalid: expect ["
      //           << it->second << "]";
      return FAIL;
    }
  } else {
    std::string proto_desc = "";
    ProtobufFactory::Instance()->GetDescriptorString(msgtype, &proto_desc);
    //if (proto_desc == "") {
      //LOG_ERROR << CYBERTRON_ERROR <<  << " channel[" << channel << "] proto_desc empty.";
      //return FAIL;
    //}
    AddChannel(channel, msgtype, proto_desc);
  }

  std::string msgstr("");
  if (!message.SerializeToString(&msgstr)) {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR << " Failed to serialize message for channel [" << channel << "]";
    return FAIL;
  }

  fueling::common::record::kinglong::proto::cybertron::SingleMsg singlemsg;
  singlemsg.set_channelname(channel);
  singlemsg.set_msg(msgstr);
  if (time > 0) {
    singlemsg.set_time(time);
  } else {
    // singlemsg.set_time(Time::Now().ToNanosecond());
  }
  return Write(singlemsg);
}

template <>
inline int DataFile::Write<MessageBase>(
    const std::string& channel,
    const std::shared_ptr<const MessageBase>& message, const uint64_t time) {
  std::string msgtype = message->_type_name;
  auto it = types_.find(channel);
  if (it != types_.end()) {
    if (it->second != msgtype) {
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_INVALID_MSG_TYPE_ERROR << " Message type [" << msgtype << "] is invalid: expect ["
                // << it->second << "]";
      return FAIL;
    }
  } else {
    std::string proto_desc = "";
    ProtobufFactory::Instance()->GetDescriptorString(msgtype, &proto_desc);
    //if (proto_desc == "") {
      //LOG_ERROR << CYBERTRON_ERROR <<  << " channel[" << channel << "] proto_desc empty.";
      //return FAIL;
    //}
    AddChannel(channel, msgtype, proto_desc);
  }

  std::string msgstr("");
  // ERROR_AND_RETURN_VAL_IF_NULL(message->_msg, FAIL, CYBERTRON_ERROR, RECORD_MSG_NULL_ERROR);
  if (!message->_msg->SerializeToString(&msgstr)) {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR << " Failed to serialize message for channel [" << channel << "]";
    return FAIL;
  }
  fueling::common::record::kinglong::proto::cybertron::SingleMsg singlemsg;
  singlemsg.set_channelname(channel);
  singlemsg.set_msg(msgstr);
  if (time > 0) {
    singlemsg.set_time(time);
  } else {
    // singlemsg.set_time(Time::Now().ToNanosecond());
  }
  return Write(std::move(singlemsg));
}

template <>
inline int DataFile::Write<RawMessage>(
    const std::string& channel,
    const std::shared_ptr<const RawMessage>& message, const uint64_t time) {
  std::string msgtype = message->get_type_name();
  auto it = types_.find(channel);
  if (it != types_.end()) {
    if (it->second != msgtype) {
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_INVALID_MSG_TYPE_ERROR << " Message type [" << msgtype << "] is invalid: expect ["
      //           << it->second << "]";
      return FAIL;
    }
  } else {
    std::string proto_desc = "";
    ProtobufFactory::Instance()->GetDescriptorString(msgtype, &proto_desc);
    //if (proto_desc == "") {
      //LOG_ERROR << CYBERTRON_ERROR <<  << " channel[" << channel << "] proto_desc empty.";
      //return FAIL;
    //}
    AddChannel(channel, msgtype, proto_desc);
  }

  // std::string msgstr("");
  // ERROR_AND_RETURN_VAL_IF_NULL(message, FAIL, CYBERTRON_ERROR, RECORD_MSG_NULL_ERROR);
  // if (!message->_msg->SerializeToString(&msgstr)) {
  //  LOG_ERROR << CYBERTRON_ERROR <<  << " Failed to serialize message for channel [" << channel <<
  // "]";
  //  return FAIL;
  //}
  fueling::common::record::kinglong::proto::cybertron::SingleMsg singlemsg;
  singlemsg.set_channelname(channel);
  singlemsg.set_msg(message->get_msg());
  if (time > 0) {
    singlemsg.set_time(time);
  } else {
    // singlemsg.set_time(Time::Now().ToNanosecond());
  }
  return Write(std::move(singlemsg));
}

}  // namespace cybertron
