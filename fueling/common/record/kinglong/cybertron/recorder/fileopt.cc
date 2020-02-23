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

#include <iostream>

#include "fueling/common/record/kinglong/cybertron/common/file_util.h"
#include "fueling/common/record/kinglong/cybertron/recorder/fileopt.h"

namespace cybertron {
#undef DO_IF
#define DO_IF(condition, code) \
  if (condition) {             \
    code                       \
  }

#if !defined(ERROR_AND_RETURN_VAL_IF)
#define ERROR_AND_RETURN_VAL_IF(condition, val, code, sub_code)         \
  if (condition) {                                                      \
    std::cout << #code << #sub_code << " " << #condition << " is met."; \
    return val;                                                         \
  }
#endif

FileOpt::FileOpt() {}
FileOpt::~FileOpt() {}

Section::Section() {}
Section::Section(const fueling::common::record::kinglong::proto::cybertron::SectionType& stype) : type(stype) {}

InFileOpt::InFileOpt() {}

InFileOpt::~InFileOpt() {}

int InFileOpt::FormatCheck(const std::string& path) {
  if (!FileUtil::Exists(path)) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_EXIST_ERROR << " file [" << path << "] not exist error.";
    return FAIL;
  }
  // get the file size
  std::ifstream ifs(path.c_str(), std::ifstream::ate | std::ifstream::binary);
  _file_size = ifs.tellg();
  if (_file_size <= OLD_HEADER_SECTION_LENGTH) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_FORMAT_ERROR << " file [" << path << "] format error.";
    return FAIL;
  }
  // LOG_INFO << "record file: " << path << ", size: " << _file_size;
  return SUCC;
}

int InFileOpt::open(const std::string& path) {
  std::lock_guard<std::mutex> lck(_mutex);
  _path = path;
  if (FormatCheck(_path) != SUCC) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_FORMAT_ERROR << " file [" << _path << "] check error.";
    return FAIL;
  }
  if (_instream.is_open()) {
    _instream.close();
  }
  std::ios_base::openmode mode = std::ios::binary | std::ios::in;
  _instream.open(_path, mode);
  // ERROR_AND_RETURN_VAL_IF_NULL(_instream, FAIL, CYBERTRON_ERROR, FILE_STREAM_INIT_ERROR);
  ERROR_AND_RETURN_VAL_IF(!_instream.is_open(), FAIL, CYBERTRON_ERROR, FILE_STREAM_INIT_ERROR);
  ERROR_AND_RETURN_VAL_IF(ReadHeaderImpl(&_header_section) != SUCC, FAIL, CYBERTRON_ERROR, READ_HEADER_ERROR);
  ERROR_AND_RETURN_VAL_IF(ReadIndexImpl(&_index_section, true) != SUCC, FAIL, CYBERTRON_ERROR, READ_INDEX_ERROR);
  return SUCC;
}

int InFileOpt::OpenWithoutHeaderAndIndex(const std::string& path) {
  std::lock_guard<std::mutex> lck(_mutex);
  _path = path;
  if (_instream.is_open()) {
    _instream.close();
  }
  if (FormatCheck(_path) != SUCC) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_FORMAT_ERROR << " file [" << _path << "] check error.";
    return FAIL;
  }
  std::ios_base::openmode mode = std::ios::binary | std::ios::in;
  _instream.open(_path, mode);
  // ERROR_AND_RETURN_VAL_IF_NULL(_instream, FAIL, CYBERTRON_ERROR, FILE_STREAM_INIT_ERROR);
  ERROR_AND_RETURN_VAL_IF(!_instream.is_open(), FAIL, CYBERTRON_ERROR, FILE_STREAM_INIT_ERROR);
  return SUCC;
}

int InFileOpt::ReadHeader() {
  ERROR_AND_RETURN_VAL_IF(ReadHeaderImpl(&_header_section) != SUCC, FAIL, CYBERTRON_ERROR, READ_HEADER_ERROR);
  return SUCC;
}

int InFileOpt::ReadIndex() {
  ERROR_AND_RETURN_VAL_IF(ReadIndexImpl(&_index_section, true) != SUCC, FAIL, CYBERTRON_ERROR, READ_INDEX_ERROR);
  return SUCC;
}

const fueling::common::record::kinglong::proto::cybertron::HeaderSection& InFileOpt::get_header() {
  return _header_section;
}

fueling::common::record::kinglong::proto::cybertron::IndexSection InFileOpt::get_index() { return _index_section; }

int InFileOpt::ReadChunk(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                         fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody) {
  if (!_instream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  std::lock_guard<std::mutex> lck(_mutex);
  return ReadChunkImplBySearch(chunkheader, chunkbody);
}

int InFileOpt::ReadIndex(fueling::common::record::kinglong::proto::cybertron::IndexSection* index) {
  if (!_instream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  std::lock_guard<std::mutex> lck(_mutex);
  return ReadIndexImpl(index);
}

int InFileOpt::ReadChunk(int chunk_idx,
                         fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                         fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody,
                         bool reset) {
  if (!_instream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  if (chunk_idx < 0) {
    // LOG_ERROR << CYBERTRON_ERROR << CHUNK_INDEX_ERROR << " param chunk_idx[" << chunk_idx << "] < 0 error.";
    return FAIL;
  }
  std::lock_guard<std::mutex> lck(_mutex);
  uint64_t pos_old = _instream.tellg();
  _instream.seekg(0, std::ios::beg);

  int ret = SUCC;
  if (_header_section.index_pos() >= HEADER_SECTION_LENGTH) {
    ret = ReadChunkIndexImpl(chunk_idx, chunkheader, chunkbody);
  } else {
    // LOG_ERROR << CYBERTRON_ERROR << HEADER_LENGHT_ERROR << " _header_section.index_pos()[" << _header_section.index_pos()
    //           << "] error.";
  }

  if (reset) {
    _instream.seekg(pos_old, std::ios::beg);
  }
  return ret;
}

int InFileOpt::ResetToChunk() {
  if (!_instream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  _instream.seekg(HEADER_SECTION_LENGTH + sizeof(Section), std::ios::beg);
  return SUCC;
}

void InFileOpt::Reset() { _instream.seekg(0, std::ios::beg); }

int InFileOpt::ReadReserve(fueling::common::record::kinglong::proto::cybertron::ReserveSection* reserve) {
  return _read_section(reserve, fueling::common::record::kinglong::proto::cybertron::RESERVE_SECTION);
}

int InFileOpt::ReadParam(fueling::common::record::kinglong::proto::cybertron::ParamSection* param) {
  return _read_section(param, fueling::common::record::kinglong::proto::cybertron::PARAM_SECTION);
}

void InFileOpt::close() {
  std::lock_guard<std::mutex> lck(_mutex);
  DO_IF (_instream.is_open(), {
    _instream.close();
  });
}

int InFileOpt::ReadHeaderImpl(fueling::common::record::kinglong::proto::cybertron::HeaderSection* header) {
  if (!_instream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  Section sec;
  _instream.read((char*)&sec, sizeof(sec));
  if (_instream.eof()) {
    _instream.clear(std::ios::goodbit);
    return TAIL;
  }
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sizeof(sec), FAIL, CYBERTRON_ERROR, FILE_NOT_OPEN_ERROR);
  std::string header_msg;
  ERROR_AND_RETURN_VAL_IF(sec.size >= _file_size || sec.size < 0, FAIL, CYBERTRON_ERROR, FILE_FORMAT_ERROR)
  header_msg.resize(sec.size);
  _instream.read((char*)header_msg.c_str(), sec.size);
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sec.size, FAIL, CYBERTRON_ERROR, FILE_NOT_OPEN_ERROR);
  // LOG_INFO << "header type:" << sec.type;
  if (sec.type == fueling::common::record::kinglong::proto::cybertron::HEADER_SECTION_TWOM) {
    _instream.seekg(HEADER_SECTION_LENGTH + sizeof(sec), std::ios::beg);
  } else {
    _instream.seekg(OLD_HEADER_SECTION_LENGTH + sizeof(sec), std::ios::beg);
  }
  if (!header->ParseFromString(header_msg)) {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " file [" << _path << "] header ParseFromString error.";
    return FAIL;
  }
  return SUCC;
}

int InFileOpt::ReadChunkImplByIndex(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                             fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody) {
  int result = ReadChunkHeaderImplByIndex(chunkheader);
  if (result != SUCC) {
    return result;
  }
  result = ReadChunkBodyImpl(chunkheader, chunkbody);
  return result;
}

int InFileOpt::ReadChunkImplBySearch(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                             fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody) {
  int result = ReadChunkHeaderImplBySearch(chunkheader);
  if (result != SUCC) {
    // LOG_WARN << "ReadChunkHeaderImplBySearch error.";
    return result;
  }
  result = ReadChunkBodyImpl(chunkheader, chunkbody);
  return result;
}

bool InFileOpt::IsEnd() { return _instream.eof(); }

int InFileOpt::ReadChunkHeaderImplByIndex(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader) {
  Section sec;
  _instream.read((char*)&sec, sizeof(sec));
  if (_instream.eof()) {
    // LOG_DEBUG << "file [" << _path << "] reach end.";
    _instream.clear(std::ios::goodbit);
    return TAIL;
  }
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sizeof(sec), FAIL, CYBERTRON_ERROR, READ_CHUNK_ERROR);
  std::string chunkheader_msg;
  ERROR_AND_RETURN_VAL_IF(sec.size >= _file_size || sec.size < 0, FAIL, CYBERTRON_ERROR, FILE_FORMAT_ERROR)
  chunkheader_msg.resize(sec.size);
  _instream.read((char*)chunkheader_msg.c_str(), sec.size);
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sec.size, FAIL, CYBERTRON_ERROR, READ_CHUNK_ERROR);
  if (!chunkheader->ParseFromString(chunkheader_msg)) {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " file [" << _path << "] chunk ParseFromString error.";
    return FAIL;
  }
  return SUCC;
}

int InFileOpt::ReadChunkHeaderImplBySearch(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader) {
  Section sec;
  do {
    _instream.read((char*)&sec, sizeof(sec));
    if (_instream.eof()) {
      // LOG_INFO << "file [" << _path << "] reach end.";
      _instream.clear(std::ios::goodbit);
      return TAIL;
    }
    ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sizeof(sec), FAIL, CYBERTRON_ERROR, READ_CHUNK_ERROR);
    if (sec.type == fueling::common::record::kinglong::proto::cybertron::CHUNK_HEADER) {
      break;
    }
    _instream.seekg(sec.size, std::ios::cur);
    if (_instream.tellg() <= 0) {
      _instream.clear(std::ios::goodbit);
      return TAIL;
    }
  } while(true);
  std::string chunkheader_msg;
  ERROR_AND_RETURN_VAL_IF(sec.size >= _file_size || sec.size < 0, FAIL, CYBERTRON_ERROR, FILE_FORMAT_ERROR);
  chunkheader_msg.resize(sec.size);
  _instream.read((char*)chunkheader_msg.c_str(), sec.size);
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sec.size, FAIL, CYBERTRON_ERROR, READ_CHUNK_ERROR);
  DO_IF (!chunkheader->ParseFromString(chunkheader_msg), {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " file [" << _path << "] chunk ParseFromString error.";
    return FAIL;
  });
  return SUCC;
}


int InFileOpt::ReadChunkBodyImpl(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                                 fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody) {
  Section sec;
  _instream.read((char*)&sec, sizeof(sec));
  if (_instream.eof()) {
    // LOG_DEBUG << "file [" << _path << "] reach end.";
    _instream.clear(std::ios::goodbit);
    return TAIL;
  }
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sizeof(sec), FAIL, CYBERTRON_ERROR, READ_CHUNK_ERROR);
  std::string chunk_msg;
  ERROR_AND_RETURN_VAL_IF(sec.size >= _file_size || sec.size < 0, FAIL, CYBERTRON_ERROR, FILE_FORMAT_ERROR)
  chunk_msg.resize(sec.size);
  _instream.read((char*)chunk_msg.c_str(), sec.size);
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sec.size, FAIL, CYBERTRON_ERROR, READ_CHUNK_ERROR);
  auto compress = CompressFactory::Create(_header_section.compress());
  if (compress == nullptr) {
    DO_IF (!chunkbody->ParseFromString(chunk_msg), {
      // LOG_WARN << "file [" << _path << "] chunk ParseFromString error.";
      return FAIL;
    });
  } else {
    std::string decompress_str;
    decompress_str.resize(chunkheader->rawsize());
    compress->Decompress(chunk_msg, decompress_str);
    DO_IF (!chunkbody->ParseFromString(decompress_str), {
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " file [" << _path << "] chunk ParseFromString error.";
      return FAIL;
    });
  }
  return SUCC;
}

int InFileOpt::ReadChunkIndexImpl(uint32_t chunk_idx,
                                  fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody) {
  std::unordered_map<int, uint64_t> chunk_map;
  for (int idx = 0; idx < _index_section.indexs_size(); ++idx) {
    if (_index_section.indexs(idx).type() == fueling::common::record::kinglong::proto::cybertron::CHUNK_HEADER) {
      chunk_map[chunk_map.size()] = _index_section.indexs(idx).pos();
    }
  }
  if (chunk_idx >= chunk_map.size()) {
    // LOG_ERROR << CYBERTRON_ERROR << CHUNK_INDEX_ERROR << " chunk_index[" << chunk_idx << "] out of range (0, "
    //           << chunk_map.size() << ")";
    return FAIL;
  }
  _instream.seekg(chunk_map[chunk_idx], std::ios::beg);
  fueling::common::record::kinglong::proto::cybertron::ChunkHeader chunkheader;
  return ReadChunkImplByIndex(&chunkheader, chunkbody);
}

int InFileOpt::ReadChunkIndexImpl(uint32_t chunk_idx,
                                  fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                                  fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody) {
  std::unordered_map<int, uint64_t> chunk_map;
  for (int idx = 0; idx < _index_section.indexs_size(); ++idx) {
    if (_index_section.indexs(idx).type() == fueling::common::record::kinglong::proto::cybertron::CHUNK_HEADER) {
      chunk_map[chunk_map.size()] = _index_section.indexs(idx).pos();
    }
  }
  if (chunk_idx >= chunk_map.size()) {
    // LOG_ERROR << CYBERTRON_ERROR << CHUNK_INDEX_ERROR << " chunk_index[" << chunk_idx << "] out of range (0, "
    //           << chunk_map.size() << ")";
    return FAIL;
  }
  _instream.seekg(chunk_map[chunk_idx], std::ios::beg);
  return ReadChunkImplByIndex(chunkheader, chunkbody);
}

int InFileOpt::ReadIndexImpl(fueling::common::record::kinglong::proto::cybertron::IndexSection* index, bool flag) {
  uint64_t pos_old = _instream.tellg();
  if (flag && _header_section.index_pos() != 0) {
    _instream.seekg(_header_section.index_pos(), std::ios::beg);
  }

  Section sec;
  _instream.read((char*)&sec, sizeof(sec));
  if (_instream.eof()) {
    _instream.clear(std::ios::goodbit);
    return TAIL;
  }
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sizeof(sec), FAIL, CYBERTRON_ERROR, READ_INDEX_ERROR);
  std::string index_msg;
  ERROR_AND_RETURN_VAL_IF(sec.size >= _file_size || sec.size < 0, FAIL, CYBERTRON_ERROR, FILE_FORMAT_ERROR)
  index_msg.resize(sec.size);
  _instream.read((char*)index_msg.c_str(), sec.size);
  ERROR_AND_RETURN_VAL_IF(_instream.gcount() != sec.size, FAIL, CYBERTRON_ERROR, READ_INDEX_ERROR);
  DO_IF (!index->ParseFromString(index_msg), {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " file [" << _path << "] index ParseFromString error.";
    return FAIL;
  });

  if (flag && _header_section.index_pos() != 0) {
    _instream.seekg(pos_old, std::ios::beg);
  }
  return SUCC;
}

OutFileOpt::OutFileOpt() {}
OutFileOpt::~OutFileOpt() { close(); }


int OutFileOpt::open(const std::string& path) {
  std::lock_guard<std::mutex> lck(_mutex);
  std::ios_base::openmode mode = std::ios::binary | std::ios::out;
  _outstream.open(path, mode);
  if (!_outstream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_FILE_OPEN_ERROR << " file [" << path << "] open error.";
    return FAIL;
  }
  _path = path;
  // LOG_DEBUG << "file[" << path << "] open succ.";
  return SUCC;
}

const fueling::common::record::kinglong::proto::cybertron::HeaderSection& OutFileOpt::get_header() {
  return _header_section;
}

void OutFileOpt::UpdateMessageNum(
    std::unordered_map<std::string, int> msg_count) {
  for (int i = 0; i < _header_section.channels_size(); ++i) {
    auto channel = _header_section.channels(i).name();
    auto it = msg_count.find(channel);
    if (it != msg_count.end()) {
      _header_section.mutable_channels(i)->set_msg_num(it->second);
      // LOG_DEBUG << "write message number with channel: [" << channel
      //           << "] num : " << msg_count[channel];
    } else {
      _header_section.mutable_channels(i)->set_msg_num(0);
      // LOG_DEBUG << "Can not find messsage number of channel [" << channel
      //           << "]";
    }
  }
  // LOG_DEBUG << "Finished update message number";
}

int OutFileOpt::get_channel_size() {
  return _header_section.channels_size();
}

void OutFileOpt::AddChannel(const std::string& channel_name,
                            const std::string& type, const std::string& desc) {
  std::lock_guard<std::mutex> lck(_mutex);
  for (int idx = 0; idx < _header_section.channels_size(); ++idx) {
    if (_header_section.channels(idx).name() == channel_name) {
      if (_header_section.channels(idx).proto_desc() != "") {
        return;
      }
      _need_write_header = true;
      _header_section.mutable_channels(idx)->set_proto_desc(desc);
      return;
    }
  }
  // LOG_DEBUG << "add new channel [" << channel_name << "] into header.";
  fueling::common::record::kinglong::proto::cybertron::Channel* channel = _header_section.add_channels();
  channel->set_name(channel_name);
  channel->set_type(type);
  channel->set_proto_desc(desc);
  channel->set_msg_num(0);
  _need_write_header = true;
}

void OutFileOpt::set_header(const fueling::common::record::kinglong::proto::cybertron::HeaderSection& header) {
  _header_section = header;
}

int OutFileOpt::WriteHeader(const fueling::common::record::kinglong::proto::cybertron::HeaderSection& header) {
  // LOG_DEBUG << "Write Header: " << std::endl << header.DebugString();
  if (!_outstream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  _header_section = header;
  _write_header_md5();

  std::lock_guard<std::mutex> lck(_mutex);
  _outstream.seekp(0, std::ios::beg);
  Section sec(fueling::common::record::kinglong::proto::cybertron::HEADER_SECTION_TWOM);
  std::string header_str;
  DO_IF (!_header_section.SerializeToString(&header_str), {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR << " file [" << _path << "] header SerializeToString error.";
    return FAIL;
  });
  if (header_str.size() > HEADER_SECTION_LENGTH) {
    // LOG_ERROR << CYBERTRON_ERROR << HEADER_LENGHT_ERROR << " header_str.size()[" << header_str.size()
    //           << "] > HEADER_SECTION_LENGTH [" << HEADER_SECTION_LENGTH
    //           << "] error.";
    return FAIL;
  }

  sec.size = header_str.size();
  _outstream.write((const char*)&sec, (int)sizeof(sec));
  _outstream.write((const char*)header_str.c_str(), header_str.size());
  static char blank[HEADER_SECTION_LENGTH] = {'0'};
  _outstream.write((const char*)blank,
                   HEADER_SECTION_LENGTH - header_str.size());

  return SUCC;
}

int OutFileOpt::WriteHeader() {
  // LOG_DEBUG << "Write Header: " << std::endl << _header_section.DebugString();
  if (!_outstream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  _write_header_md5();

  std::lock_guard<std::mutex> lck(_mutex);
  _outstream.seekp(0, std::ios::beg);
  Section sec(fueling::common::record::kinglong::proto::cybertron::HEADER_SECTION_TWOM);
  std::string header_str;
  DO_IF (!_header_section.SerializeToString(&header_str), {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR << " file [" << _path << "] header SerializeToString error.";
    return FAIL;
  });
  if (header_str.size() > HEADER_SECTION_LENGTH) {
    // LOG_ERROR << CYBERTRON_ERROR << HEADER_LENGHT_ERROR << " header_str.size()[" << header_str.size()
    //           << "] > HEADER_SECTION_LENGTH [" << HEADER_SECTION_LENGTH
    //           << "] error.";
    return FAIL;
  }
  sec.size = header_str.size();

  _outstream.write((const char*)&sec, (int)sizeof(sec));
  _outstream.write((const char*)header_str.c_str(), header_str.size());
  static char blank[HEADER_SECTION_LENGTH] = {'0'};
  _outstream.write((const char*)blank,
                   HEADER_SECTION_LENGTH - header_str.size());
  _outstream.flush();
  return SUCC;
}

int OutFileOpt::WriteChunk(fueling::common::record::kinglong::proto::cybertron::ChunkHeader& chunkheader,
                           const fueling::common::record::kinglong::proto::cybertron::ChunkSection& chunkbody) {
  if (!_outstream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }

  if (_header_section.begintime() == 0) {
    _header_section.set_begintime(chunkheader.begintime());
  }
  if (_need_write_header) {
    uint64_t pos_old = _outstream.tellp();
    _outstream.seekp(0, std::ios::beg);

    DO_IF (WriteHeader(_header_section) != SUCC, {
      // LOG_ERROR << CYBERTRON_ERROR << WRITE_HEADER_ERROR << " Write Header error.";
    });
    _outstream.seekp(pos_old, std::ios::beg);
    _need_write_header = false;
  }
  std::lock_guard<std::mutex> lck(_mutex);
  return WriteChunkImpl(chunkheader, chunkbody);
}

int OutFileOpt::WriteReserve(const fueling::common::record::kinglong::proto::cybertron::ReserveSection& reserve) {
  if (!_outstream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  std::lock_guard<std::mutex> lck(_mutex);
  Section sec(fueling::common::record::kinglong::proto::cybertron::RESERVE_SECTION);
  std::string reserve_str;
  DO_IF (!reserve.SerializeToString(&reserve_str), {
    return FAIL;
  });
  sec.size = reserve_str.size();

  fueling::common::record::kinglong::proto::cybertron::Index* index = _index_section.add_indexs();
  index->set_type(sec.type);
  index->set_pos(_outstream.tellp());

  _outstream.write((const char*)&sec, (int)sizeof(sec));
  _outstream.write((const char*)reserve_str.c_str(), reserve_str.size());
  return SUCC;
}

int OutFileOpt::WriteParam(const fueling::common::record::kinglong::proto::cybertron::ParamSection& param) {
  if (!_outstream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  std::lock_guard<std::mutex> lck(_mutex);
  Section sec(fueling::common::record::kinglong::proto::cybertron::PARAM_SECTION);
  std::string param_str;
  DO_IF (!param.SerializeToString(&param_str), {
    return FAIL;
  });
  sec.size = param_str.size();

  fueling::common::record::kinglong::proto::cybertron::Index* index = _index_section.add_indexs();
  index->set_type(sec.type);
  index->set_pos(_outstream.tellp());

  _outstream.write((const char*)&sec, (int)sizeof(sec));
  _outstream.write((const char*)param_str.c_str(), param_str.size());
  return SUCC;
}

int OutFileOpt::WriteIndex(const fueling::common::record::kinglong::proto::cybertron::IndexSection& index) {
  if (!_outstream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
    return FAIL;
  }
  std::lock_guard<std::mutex> lck(_mutex);
  _header_section.set_index_pos(_outstream.tellp());
  Section sec(fueling::common::record::kinglong::proto::cybertron::INDEX_SECTION);
  std::string index_str;
  DO_IF (!index.SerializeToString(&index_str), {
    // LOG_ERROR << CYBERTRON_ERROR << "Index serialize to string failed";
    return FAIL;
  });
  sec.size = index_str.size();
  _outstream.write((const char*)&sec, (int)sizeof(sec));
  _outstream.write((const char*)index_str.c_str(), index_str.size());
  _header_section.set_size(_outstream.tellp());
  return SUCC;
}

uint64_t OutFileOpt::get_file_size() { return _outstream.tellp(); }

void OutFileOpt::close() {
  if (!_outstream.is_open()) {
    return;
  }

  try {
    WriteIndex(_index_section);
    // LOG_DEBUG << "fileopt writing header:" << _header_section.DebugString();
    _header_section.set_finish(fueling::common::record::kinglong::proto::cybertron::RecordStatus::FINISH);
    WriteHeader(_header_section);
  }
  catch (std::exception& e) {
    // LOG_ERROR << CYBERTRON_ERROR << WRITE_EXCEPTION_ERROR << " fileopt write with exception : " << e.what();
  }

  _outstream.close();
  // LOG_DEBUG << "OutFile closed.";
}

void OutFileOpt::_write_header_md5() {
  std::string md5_raw_str;
  std::ostringstream oss;
  md5_raw_str += ("path" + _header_section.path());
  md5_raw_str += ("version" + _header_section.version());
  oss << _header_section.compress();
  md5_raw_str += ("compress" + std::string(oss.str()));
  oss.str("");
  oss << _header_section.chunk_interval();
  md5_raw_str += ("chunk_interval" + std::string(oss.str()));
  oss.str("");
  md5_raw_str += "channel";
  for (int i = 0; i < _header_section.channels_size(); ++i) {
    md5_raw_str += _header_section.channels(i).name();
    md5_raw_str += _header_section.channels(i).type();
    oss << _header_section.channels(i).msg_num();
    md5_raw_str += std::string(oss.str());
    oss.str("");
  }
  oss << _header_section.index_pos();
  md5_raw_str += ("index_pos" + std::string(oss.str()));
  oss.str("");
  oss << _header_section.chunknum();
  md5_raw_str += ("chunknum" + std::string(oss.str()));
  oss.str("");
  oss << _header_section.begintime();
  md5_raw_str += ("begintime" + std::string(oss.str()));
  oss.str("");
  oss << _header_section.endtime();
  md5_raw_str += ("endtime" + std::string(oss.str()));
  oss.str("");
  oss << _header_section.msgnum();
  md5_raw_str += ("msgnum" + std::string(oss.str()));
  oss.str("");
  oss << _header_section.size();
  md5_raw_str += ("size" + std::string(oss.str()));

  MD5_cal md5;
  std::string md5_str = md5.digestString(md5_raw_str.c_str());
  _header_section.set_md5(md5_str);
}

int OutFileOpt::WriteChunkImpl(fueling::common::record::kinglong::proto::cybertron::ChunkHeader& chunkheader,
                               const fueling::common::record::kinglong::proto::cybertron::ChunkSection& chunk) {
  // LOG_DEBUG << "start write chunk:\n" << chunkheader.DebugString();
  std::string chunk_str;
  DO_IF (!chunk.SerializeToString(&chunk_str), {
    // LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR << " chunk.SerializeToString error.";
    return FAIL;
  });
  chunkheader.set_rawsize(chunk_str.size());
  {
    _header_section.set_chunknum(_header_section.chunknum() + 1);
    if (_header_section.begintime() == 0) {
      _header_section.set_begintime(chunkheader.begintime());
    }

    if (_header_section.endtime() > chunkheader.endtime()) {
      // LOG_WARN << "Invalid end time:" << chunkheader.endtime();
    } else {
      _header_section.set_endtime(chunkheader.endtime());
    }

    _header_section.set_msgnum(_header_section.msgnum() + chunkheader.msgnum());

    Section sec(fueling::common::record::kinglong::proto::cybertron::CHUNK_HEADER);
    std::string chunkheader_str;
    DO_IF (!chunkheader.SerializeToString(&chunkheader_str), {
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_SERIALIZE_STR_ERROR << " chunkheader.SerializeToString error.";
      return FAIL;
    });
    sec.size = chunkheader_str.size();

    fueling::common::record::kinglong::proto::cybertron::Index* index = _index_section.add_indexs();
    index->set_type(sec.type);
    index->set_pos(_outstream.tellp());

    _outstream.write((const char*)&sec, (int)sizeof(sec));
    _outstream.write((const char*)chunkheader_str.c_str(),
                     chunkheader_str.size());
  }

  {
    Section sec(fueling::common::record::kinglong::proto::cybertron::CHUNK_SECTION);
    std::string compress_str;
    auto compress = CompressFactory::Create(_header_section.compress());
    if (compress == nullptr) {
      //LOG_ERROR << CYBERTRON_ERROR <<  << " compress type[" << _header_section.compress() << "] error";
      sec.size = chunk_str.size();
      fueling::common::record::kinglong::proto::cybertron::Index* index = _index_section.add_indexs();
      index->set_type(sec.type);
      index->set_pos(_outstream.tellp());
      _outstream.write((const char*)&sec, (int)sizeof(sec));
      _outstream.write((const char*)chunk_str.c_str(), chunk_str.size());
    } else {
      compress->compress(chunk_str, compress_str);
      sec.size = compress_str.size();
      fueling::common::record::kinglong::proto::cybertron::Index* index = _index_section.add_indexs();
      index->set_type(sec.type);
      index->set_pos(_outstream.tellp());
      _outstream.write((const char*)&sec, (int)sizeof(sec));
      _outstream.write((const char*)compress_str.c_str(), compress_str.size());
    }
  }

  // LOG_DEBUG << "finish write chunk";
  _outstream.flush();
  return SUCC;
}
#undef DO_IF

}  // namespace cybertron
