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

#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <iostream>
#include <memory>
#include <vector>
#include <thread>

#include "fueling/common/record/kinglong/cybertron/recorder/py_message_shifter.h"
#include "fueling/common/record/kinglong/cybertron/recorder/datafile.h"
#include "fueling/common/record/kinglong/cybertron/recorder/data_iterator.h"
#include "fueling/common/record/kinglong/cybertron/common/protobuf_factory.h"

#include "fueling/common/record/kinglong/proto/cybertron/cyber_header.pb.h"
#include "fueling/common/record/kinglong/proto/cybertron/record.pb.h"

namespace cybertron {

struct bag_message {
  uint64_t timestamp = 0;
  std::string topic = "";
  std::string data = "";
  std::string data_type = "";
  bool end = true;
};

class BagMessage {
 public:
  BagMessage(const std::string &data)
      : _msg(new cybertron::PyMessageShifter(data, "py_proto")) {}

  std::shared_ptr<cybertron::PyMessageShifter> _msg;
};

class PyBag {
 public:
  PyBag(const std::string &file_name, bool write_mode = false, bool if_dump_parameter_snapshot = false)
      : _file_name(file_name), _data_file(new cybertron::DataFile()) {
    if (write_mode) {
      _status = _data_file->open(file_name, FileMode::Write, if_dump_parameter_snapshot);
    } else {
      _status = _data_file->open(file_name, FileMode::Read, if_dump_parameter_snapshot);
    }
    //_data_file->Init();
  }

  ~PyBag() {}

  void reset() { _data_it.reset(); }

  bool is_valid() {
    return _status == 0;
  }

  bag_message read(const std::vector<std::string> &topics, uint64_t start = 0,
                   uint64_t end = ULONG_MAX) {
    bag_message m;

    // add data iterator
    if (!_data_it) {
      _data_it = std::make_shared<DataIterator>(_data_file);

      if (topics.size() > 0) {
        BaseFilter::SharedPtr cf = std::make_shared<ChannelFilter>(topics);
        _data_it->AddFilter(cf);
        // std::cout << "channel filter size: " << topics.size() << std::endl;
      }
      if (end == 0) {
        end = ULONG_MAX;
      }
#if 0
      for (size_t i = 0; i < topics.size(); ++i) {
        BaseFilter::SharedPtr cf = std::make_shared<ChannelFilter>(topics[i]);
        _data_it->add_filter(cf);
        //std::cout << "topic: " << topics[i] << std::endl;
      }
#endif
      if ((start < end) || (start && (start == end))) {
        BaseFilter::SharedPtr cf = std::make_shared<TimerFilter>(start, end);
        _data_it->AddFilter(cf);
      }
      _data_it_pos.reset(new DataIterator::iterator(_data_it.get()));
    }

    if (*_data_it_pos.get() == _data_it->end()) {
      _data_it.reset();
      m.end = true;
    } else {
      m.end = false;
      m.topic = (*_data_it_pos.get())->channelname();
      m.data = (*_data_it_pos.get())->msg();
      m.timestamp = (*_data_it_pos.get())->time();
      m.data_type = _data_file->get_channel_type(m.topic);
      (*_data_it_pos.get())++;
    }

    return m;
  }

  void register_message(const std::string &desc) {
    ProtobufFactory::Instance()->RegisterPythonMessage(desc);
  }

  std::string get_desc(const std::string& name) {
    return _data_file->GetDescByName(name);
  }

  void set_desc(const std::string& name, const std::string& type, const std::string& desc) {
    _data_file->AddChannel(name, type, desc);
  }

  std::string get_snapshot() {
    return _data_file->GetSnapshot();
  }

  void set_snapshot(const std::string& snapshot) {
    _data_file->SetSnapshot(snapshot);
  }

  bool write(const std::string &channel, const std::string &data,
             const std::string &data_type, uint64_t timestamp = 0) {
    const std::shared_ptr<const cybertron::PyMessageShifter> message(
        new cybertron::PyMessageShifter(data, data_type));

    _data_file->Write(channel, message, timestamp);
    return true;
  }

  uint32_t get_message_count(const std::string &channel = "") {
    if (channel == "") {
      return _data_file->get_msg_num();
    } else {
      return _data_file->get_msg_num(channel);
    }
  }

  uint64_t get_start_time(const std::string &topic = "") {
    return _data_file->get_begin_time();
  }

  bool is_active() {
    return _data_file->IsActive();
  }

  uint64_t get_end_time(const std::string &topic = "") {
    return _data_file->get_end_time();
  }

  uint64_t get_file_size() {
    uint64_t file_size = _data_file->get_file_size();
    // std::cout << "File size: " << file_size << std::endl;
    return file_size;
  }

  std::string get_file_name() { return _data_file->get_file_name(); }

  std::vector<std::string> get_channels() const {
    return _data_file->get_channels();
  }

  std::string get_channel_type(const std::string &channel) const {
    return _data_file->get_channel_type(channel);
  }

  std::string get_version() const { return _data_file->get_version(); }

  int get_compress_type() const {
    int type = 0;
    switch (_data_file->get_compress_type()) {
      case fueling::common::record::kinglong::proto::cybertron::COMPRESS_NONE:
        type = 0;
        break;
      case fueling::common::record::kinglong::proto::cybertron::COMPRESS_BZ2:
        type = 1;
        break;
      default:
        type = 0;
        break;

    }
    return type;
  }

  int close() {
    _data_file->close();
    reset();
    _data_file.reset();
    //TODO: more check
    return cybertron::SUCC;
  }

 private:
  std::string _file_name;

  int _status = 1;
  std::shared_ptr<DataFile> _data_file;
  std::shared_ptr<DataIterator> _data_it;
  std::shared_ptr<DataIterator::iterator> _data_it_pos;
};

}  // namespace cybertron
