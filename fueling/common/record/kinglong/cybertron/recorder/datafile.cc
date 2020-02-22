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

#include "cybertron/recorder/datafile.h"
#include <boost/exception/diagnostic_information.hpp>
#include "cybertron/simulator/simulator.h"

namespace cybertron {
#undef DO_IF
#define DO_IF(condition, code) \
  if (condition) {             \
    code                       \
  }

DataFile::DataFile() {}

DataFile::~DataFile() {
  try {
    close();
  } catch (const boost::exception& e) {
    LOG_INFO << "close exception:" << boost::diagnostic_information(e).c_str();
  }
}

DataFile::DataFile(const std::string& filename, const uint32_t& mode, bool if_dump_parameter_snapshot) {
  int result = open(filename, mode, if_dump_parameter_snapshot);
  file_is_open_ = (result == SUCC) ? true : false;
  READ_CONF_WITH_DEFAULT("mainboard", "parameter_shot_open", parameter_shot_open_, true);
}

DataFile::DataFile(const RecorderParam& param) {
  chunk_interval_ = param.chunk_interval;
  chunk_limit_ = param.chunk_limit;
  segment_interval_ = param.segment_interval;
  compress_type_ = param.compress_type;
  version_ = param.version;
  path_ = param.path;
  READ_CONF_WITH_DEFAULT("mainboard", "parameter_shot_open", parameter_shot_open_, true);

  int result = open(param.path, FileMode::Write, true);
  file_is_open_ = (result == SUCC) ? true : false;
}

int DataFile::open(const std::string& filename, const uint32_t& mode, bool if_dump_parameter_snapshot) {
  is_flushing_ = false;
  if (simulator::Simulation::Instance()->RunModel() == "SIM") {
    run_in_sim_ = true;
  }

  if (segment_interval_ != 0 && chunk_interval_ > segment_interval_) {
    chunk_interval_ = segment_interval_;
  }

  file_name_ = filename;
  path_ = file_name_;

  ClearAll();
  file_index_ = 0;
  chunk_.reset(new Chunk());
  chunk_backup_.reset(new Chunk());
  infileopt_.reset(new InFileOpt());
  outfileopt_.reset(new OutFileOpt);

  if (mode == FileMode::Write) {
    if (segment_interval_ > 0) {
      path_ = file_name_ + "." + std::to_string(file_index_++);
      header_.set_path(path_);
    }

    if (outfileopt_->open(path_) != SUCC) {
      LOG_ERROR << CYBERTRON_ERROR << RECORD_FILE_OPEN_ERROR << " open outfile failed. filename: " << path_;
      return FAIL;
    }
    outfileopt_->WriteHeader(header_);
    if (if_dump_parameter_snapshot) {
      WriteSnapshot(if_dump_parameter_snapshot);
    }
    is_writing_ = true;
    // start flush thread
    flush_thread_ = std::make_shared<std::thread>([this]() {
        this->FlushChunk();
        });
    ERROR_AND_RETURN_VAL_IF_NULL(flush_thread_, FAIL, CYBERTRON_ERROR, FLUSH_THREAD_INIT_ERROR);
  }

  if (mode == FileMode::Read) {
    if (infileopt_->open(filename)) {
      LOG_ERROR << CYBERTRON_ERROR << RECORD_FILE_OPEN_ERROR << " open infile failed. filename: " << filename;
      return FAIL;
    }
    header_ = infileopt_->get_header();
    for (auto& channel : header_.channels()) {
      types_[channel.name()] = channel.type();
      msg_count_[channel.name()] = channel.msg_num();
      ProtobufFactory::Instance()->RegisterMessage(channel.proto_desc());
    }
    ReadSnapshot();
  }

  return SUCC;
}

void DataFile::ClearAll() {
  msg_count_.clear();
  types_.clear();

  header_.set_version(version_);
  header_.set_compress(compress_type_);
  header_.set_chunk_interval(chunk_interval_);
  header_.set_path(path_);
  header_.set_index_pos(0);
  header_.set_chunknum(0);
  header_.set_begintime(0);
  header_.set_endtime(0);
  header_.set_msgnum(0);
  header_.set_size(0);
  header_.set_md5("null");
  for (int i = 0; i < header_.channels_size(); ++i) {
    header_.mutable_channels(i)->set_msg_num(0);
  }
}

bool DataFile::IsActive() const {
  return (header_.finish() == cybertron::proto::RecordStatus::ACTIVE);
}

void DataFile::SplitOutfile(bool if_dump_parameter_snapshot) {
  outfileopt_.reset(new OutFileOpt);
  if (segment_interval_ > 0) {
    path_ = file_name_ + "." + std::to_string(file_index_++);
  }
  ClearAll();
  outfileopt_->open(path_);
  outfileopt_->WriteHeader(header_);
  if (if_dump_parameter_snapshot) {
    WriteSnapshot(if_dump_parameter_snapshot);
  }
  LOG_INFO << "split to new file: " << path_;
}

void DataFile::AddChannel(const std::string& channel_name,
                          const std::string& type,
                          const std::string& proto_desc) {
  types_[channel_name] = type;
  outfileopt_->AddChannel(channel_name, type, proto_desc);
}

std::string DataFile::get_channel_type(const std::string& channel) const {
  auto it = types_.find(channel);
  RETURN_VAL_IF2(it == types_.end(), "");
  return it->second;
}

int DataFile::Write(const cybertron::proto::SingleMsg& singlemsg) {
  std::lock_guard<std::recursive_mutex> lock(chunk_mutex_);
  DO_IF (chunk_->write(singlemsg) != SUCC, {
    return FAIL;
  });

  std::string channel = singlemsg.channelname();
  auto it = msg_count_.find(channel);
  if (it != msg_count_.end()) {
    it->second++;
  } else {
    msg_count_.insert(std::make_pair(channel, 1));
  }

  uint64_t time_diff = singlemsg.time() - chunk_->BeginTime();
  if (time_diff > chunk_interval_ || chunk_->RawSize() > chunk_limit_) {
    RETURN_VAL_IF2(flush() != SUCC, FAIL);
    chunk_->reset();
  }
  return SUCC;
}

void DataFile::set_header(const cybertron::proto::HeaderSection& header) {
  header_ = header;
  outfileopt_->set_header(header);
}

void DataFile::close() {
  if (is_writing_) {
    StopWrite();
    is_writing_ = false;
    flush_cond_.notify_all();
  }
  if (flush_thread_ && flush_thread_->joinable()) {
    LOG_DEBUG << "join flush thread.";
    flush_thread_->join();
    flush_thread_ = nullptr;
  }
  if (snapshot_ != nullptr) {
    delete snapshot_;
    snapshot_ = nullptr;
  }
}

void DataFile::StopWrite() {
  std::lock_guard<std::recursive_mutex> lock(chunk_mutex_);
  flush();
  // wait for flush finished
  while (chunk_backup_->RawSize() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  chunk_->reset();
  outfileopt_->close();
}

int DataFile::flush() {
  std::lock_guard<std::recursive_mutex> lock(chunk_mutex_);
  if (chunk_->chunk_section_.msgs_size() == 0) {
    LOG_INFO << "chunkbody is empty";
    return SUCC;
  }
  {
    std::unique_lock<std::mutex> flush_lock(flush_mutex_);
    DO_IF(run_in_sim_, {
      while (is_flushing_ && is_writing_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
    outfileopt_->UpdateMessageNum(msg_count_);
    chunk_backup_.swap(chunk_);
    if (segment_interval_ > 0) {
      if (get_begin_time() != 0 &&
          (chunk_backup_->chunk_header_.endtime() - get_begin_time() > segment_interval_)) {
        outfileopt_backup_.swap(outfileopt_);
        SplitOutfile();
      }
    }
    DO_IF(!run_in_sim_, {
      flush_cond_.notify_one();
    });
  }
  DO_IF(run_in_sim_, {
    is_flushing_ = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  });
  return SUCC;
}

void DataFile::FlushChunk() {
  LOG_INFO << "start flush chunk thread.";
  while (is_writing_) {
    DO_IF(run_in_sim_, {
      while (!is_flushing_ && is_writing_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
    std::unique_lock<std::mutex> flush_lock(flush_mutex_);
    DO_IF(!run_in_sim_, {
      flush_cond_.wait(flush_lock, [this] {
          return (chunk_backup_->RawSize() > 0) || !is_writing_;
          });
      if (!is_writing_) {
        break;
      }
    });

    DO_IF (chunk_backup_->RawSize() > 0, {
      LOG_DEBUG << "flush chunk size: " << chunk_backup_->RawSize()
        << ", begin time: " << chunk_backup_->BeginTime();
      if (segment_interval_ > 0 && outfileopt_backup_ != nullptr) {
        outfileopt_backup_->WriteChunk(chunk_backup_->chunk_header_, chunk_backup_->chunk_section_);
        outfileopt_backup_.reset(nullptr);
      } else {
        outfileopt_->WriteChunk(chunk_backup_->chunk_header_, chunk_backup_->chunk_section_);
      }
      chunk_backup_->reset();
    });

    DO_IF(run_in_sim_, {
      is_flushing_ = false;
    });
  }
  LOG_INFO << "finish flush chunk thread.";
}

void DataFile::set_chunk_interval(const uint64_t& interval) {
  chunk_interval_ = interval;
}

std::vector<std::string> DataFile::get_channels() const {
  std::vector<std::string> channels;
  for (auto& it : msg_count_) {
    channels.push_back(it.first);
  }
  return channels;
}

int DataFile::get_msg_num(const std::string& channel) const {
  auto it = msg_count_.find(channel);
  if (it != msg_count_.end()) {
    return it->second;
  }
  return 0;
}

int DataFile::get_chunk_num() const {
  cybertron::proto::ChunkHeader chunkheader;
  cybertron::proto::ChunkSection chunk;
  auto header = infileopt_->get_header();
  return header.chunknum();
}

int DataFile::get_msg_num(const int& chunk_index) const {
  cybertron::proto::ChunkHeader chunkheader;
  cybertron::proto::ChunkSection chunk;
  infileopt_->ReadChunk(chunk_index, &chunkheader, &chunk);
  return chunk.msgs_size();
}

int DataFile::get_msg_num() const {
  int msg_num = 0;
  for (auto& it : msg_count_) {
    msg_num += it.second;
  }
  return msg_num;
}

cybertron::proto::SingleMsg DataFile::ReadMessage(const int& chunk_index,
                                                  const int& msg_index) {
  cybertron::proto::ChunkHeader chunkheader;
  cybertron::proto::ChunkSection chunk;
  infileopt_->ReadChunk(chunk_index, &chunkheader, &chunk);
  return chunk.msgs(msg_index);
}

int DataFile::ReadChunk(cybertron::proto::ChunkSection* chunk,
                        cybertron::proto::ChunkHeader* header) {
  return infileopt_->ReadChunk(header, chunk);
}

void DataFile::ReadChunk(const int& index,
                         cybertron::proto::ChunkSection* chunk,
                         cybertron::proto::ChunkHeader* header, bool reset) {
  infileopt_->ReadChunk(index, header, chunk, reset);
}

void DataFile::set_chunk_limit(const uint64_t& limit) { chunk_limit_ = limit; }

void DataFile::set_segment_interval(const uint64_t& interval) {
  segment_interval_ = interval;
}

void DataFile::set_compress_type(const cybertron::proto::CompressType& type) {
  compress_type_ = type;
}

uint64_t DataFile::get_begin_time() const {
  if (is_writing_) {
    return outfileopt_->get_header().begintime();
  }
  return header_.begintime();
}

uint64_t DataFile::get_end_time() const {
  if (is_writing_) {
    return outfileopt_->get_header().endtime();
  }
  return header_.endtime();
}

std::string DataFile::GetSnapshot() {
  if (!parameter_shot_open_) {
    return "";
  }
  if (snapshot_ == nullptr) {
    //param_helper_ = ParameterRecorderHelper::make_shared();
    //snapshot_ = new proto::ParamSnapshot;
    //if (param_helper_->GetSnapshotFromParameterServer(snapshot_) != SUCC) {
    //  LOG_ERROR << CYBERTRON_ERROR <<  << " get snapshot failed.";
    //  return "";
    //}
    return "";
  }

  //ParameterRecorderHelper::GetSnapshot(param_events_, snapshot_);
  std::string param_str = "";
  DO_IF (!snapshot_->SerializeToString(&param_str), {
    LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR << " snapshot_->SerializeToString error.";
    return "";
  });
  return param_str;
}

int DataFile::SetSnapshot(const std::string& snapshot) {
  if (!parameter_shot_open_) {
    return SUCC;
  }
  if (!snapshot.empty()) {
    cybertron::proto::ParamSection param;
    param.set_paramdump(snapshot);
    outfileopt_->WriteParam(param);
  }
  if (snapshot_ == nullptr) {
    snapshot_ = new proto::ParamSnapshot;
  }
  RETURN_VAL_IF_NULL2(snapshot_, FAIL);
  if (!snapshot_->ParseFromString(snapshot)) {
    LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " snapshot_->ParseFromString error.";
    return FAIL;
  }
  return SUCC;
}

int DataFile::ReadSnapshot() {
  if (!parameter_shot_open_) {
    return SUCC;
  }
  if (snapshot_ == nullptr) {
    snapshot_ = new proto::ParamSnapshot;
  }
  RETURN_VAL_IF_NULL2(snapshot_, FAIL);
  proto::ParamSection param_section;
  if (infileopt_->ReadParam(&param_section) != SUCC) {
    //LOG_INFO << "infileopt_ ReadParam  error.";
    return FAIL;
  }
  DO_IF (!snapshot_->ParseFromString(param_section.paramdump()), {
    LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " snapshot_->ParseFromString error.";
    return FAIL;
  });
  return SUCC;
}

int DataFile::WriteSnapshot(bool if_dump_parameter_snapshot) {
  if (!parameter_shot_open_) {
    return SUCC;
  }
  if (snapshot_ != nullptr) {
    delete snapshot_;
    snapshot_ = nullptr;
  }
  snapshot_ = new proto::ParamSnapshot;
  if (if_dump_parameter_snapshot) {
    if (param_helper_ == nullptr) {
      try {
        param_helper_ = ParameterRecorderHelper::make_shared();
      } catch (const boost::exception& e) {
        LOG_INFO << "param helperexception:" << boost::diagnostic_information(e).c_str();
        return FAIL;
      }
    }
    if (param_helper_->GetSnapshotFromParameterServer(snapshot_) != SUCC) {
      LOG_WARN << "get snapshot failed.";
      return FAIL;
    }
  }

  ParameterRecorderHelper::GetSnapshot(param_events_, snapshot_);
  std::string param_str = "";
  DO_IF (!snapshot_->SerializeToString(&param_str), {
    LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR << " snapshot_->SerializeToString error.";
    return FAIL;
  });
  DO_IF (!param_str.empty(), {
    cybertron::proto::ParamSection param;
    param.set_paramdump(param_str);
    outfileopt_->WriteParam(param);
  });
  return SUCC;
}

uint64_t DataFile::get_file_size() const {
  if (is_writing_) {
    return outfileopt_->get_file_size();
  } else {
    return header_.size();
  }
}

std::string DataFile::get_file_name() const {
  if (is_writing_ && segment_interval_ > 0) {
    return (file_name_ + "." + std::to_string(file_index_ - 1));
  } else {
    return file_name_;
  }
}

int DataFile::Write(const std::string& channel, const std::string& message,
                    const std::string& type, const uint64_t time) {
  LOG_DEBUG << "Received message from channel [" << channel << "]";
  auto it = types_.find(channel);
  if (it != types_.end()) {
    DO_IF (it->second != type, {
      LOG_ERROR << CYBERTRON_ERROR << RECORD_INVALID_MSG_TYPE_ERROR << " Message type [" << type << "] is invalid: expect ["
                << it->second << "]";
      return FAIL;
    });
  } else {
    std::string proto_desc = "";
    ProtobufFactory::Instance()->GetDescriptorString(type, &proto_desc);
    if (proto_desc == "") {
      //LOG_ERROR << CYBERTRON_ERROR <<  << " channel[" << channel << "] proto_desc empty.";
      //return FAIL;
    }
    AddChannel(channel, type, proto_desc);
  }

  cybertron::proto::SingleMsg singlemsg;
  singlemsg.set_channelname(channel);
  singlemsg.set_msg(message);
  if (time > 0) {
    singlemsg.set_time(time);
  } else {
    singlemsg.set_time(Time::Now().ToNanosecond());
  }

  return Write(singlemsg);
}

std::string DataFile::GetDescByName(const std::string& name) const {
  for (auto& channel : header_.channels()) {
    DO_IF (channel.name() == name, {
      return channel.proto_desc();
    });
  }
  return "";
}

void DataFile::ShowProgress() {
  // display progress
  static int total = 0;
  std::cout << "\r[RUNNING]  Record : "
            << "    total channel num : "
            << outfileopt_->get_channel_size()
            << "  total msg num : " << ++total;
  std::cout.flush();
}
#undef DO_IF

}  // namespace cybertron
