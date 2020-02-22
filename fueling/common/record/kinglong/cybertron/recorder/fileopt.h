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

#ifndef INCLUDE_CYBERTRON_RECORDER_FILEOPT_H_
#define INCLUDE_CYBERTRON_RECORDER_FILEOPT_H_

#include <iostream>
#include <fstream>
#include <mutex>
#include <vector>
#include <unordered_map>

#include "cybertron/common/macros.h"
#include "cybertron/common/md5.h"
#include "cybertron/proto/record.pb.h"
#include "cybertron/recorder/compress.h"

namespace cybertron {

const int OLD_HEADER_SECTION_LENGTH = 204800;
const int HEADER_SECTION_LENGTH = 2048000;

class FileOpt {
 public:
  FileOpt();
  virtual ~FileOpt();
  virtual int open(const std::string& path) = 0;
  virtual void close() = 0;

  std::string _path;
  std::mutex _mutex;
};

struct Section {
  Section();
  Section(const cybertron::proto::SectionType& stype);
  int64_t size;
  cybertron::proto::SectionType type;
};

class InFileOpt : public FileOpt {
 public:
  SMART_PTR_DEFINITIONS(InFileOpt)
  InFileOpt();
  virtual ~InFileOpt();
  int open(const std::string& path) override;
  void Reset();
  void close() override;

  int OpenWithoutHeaderAndIndex(const std::string& path);
  const cybertron::proto::HeaderSection& get_header();
  cybertron::proto::IndexSection get_index();
  int ReadChunk(cybertron::proto::ChunkHeader* chunkheader,
                cybertron::proto::ChunkSection* chunkbody);
  int ReadIndex(cybertron::proto::IndexSection* index);
  // int ReadChunk(int chunk_idx, cybertron::proto::ChunkSection* chunkbody);
  int ResetToChunk();
  int ReadChunk(int chunk_idx, cybertron::proto::ChunkHeader* chunkheader,
                cybertron::proto::ChunkSection* chunkbody, bool reset = true);
  int ReadReserve(cybertron::proto::ReserveSection* reserve);
  int ReadParam(cybertron::proto::ParamSection* param);
  int ReadIndex();
  int ReadHeader();
  bool IsEnd();

 private:
  int FormatCheck(const std::string& path);
  int ReadHeaderImpl(cybertron::proto::HeaderSection* header);
  int ReadChunkImplByIndex(cybertron::proto::ChunkHeader* chunkheader,
                    cybertron::proto::ChunkSection* chunkbody);
  int ReadChunkImplBySearch(cybertron::proto::ChunkHeader* chunkheader,
                    cybertron::proto::ChunkSection* chunkbody);
  int ReadChunkHeaderImplByIndex(cybertron::proto::ChunkHeader* chunkheader);
  int ReadChunkHeaderImplBySearch(cybertron::proto::ChunkHeader* chunkheader);
  int ReadChunkBodyImpl(cybertron::proto::ChunkHeader* chunkheader,
                        cybertron::proto::ChunkSection* chunkbody);
  int ReadChunkIndexImpl(uint32_t chunk_idx,
                         cybertron::proto::ChunkSection* chunkbody);
  int ReadChunkIndexImpl(uint32_t chunk_idx,
                         cybertron::proto::ChunkHeader* chunkheader,
                         cybertron::proto::ChunkSection* chunkbody);
  int ReadIndexImpl(cybertron::proto::IndexSection* index, bool flag = false);
  template <typename T>
  int _read_section(T* section,
                    const cybertron::proto::SectionType& sectiontype) {
    if (!_instream || !_instream.is_open()) {
      LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
      return FAIL;
    }
    std::lock_guard<std::mutex> lck(_mutex);
    int pos_old = _instream.tellg();
    int pos_new = 0;
    for (int idx = 0; idx < _index_section.indexs_size(); ++idx) {
      if (_index_section.indexs(idx).type() == sectiontype) {
        pos_new = _index_section.indexs(idx).pos();
        break;
      }
    }
    if (pos_new == 0) {
      LOG_INFO << "NOTE: there is no ["
                << cybertron::proto::SectionType_Name(sectiontype)
                << "] .";
      return FAIL;
    }
    _instream.seekg(pos_new, std::ios::beg);

    Section sec;
    _instream.read((char*)&sec, sizeof(sec));
    if (_instream.eof()) {
      _instream.clear(std::ios::goodbit);
      return TAIL;
    }
    if (_instream.gcount() != sizeof(sec)) {
      LOG_ERROR << CYBERTRON_ERROR << RECORD_FILE_READ_ERROR << " file [" << _path << "] read error.";
      return FAIL;
    }
    std::string section_msg;
    section_msg.resize(sec.size);
    _instream.read((char*)section_msg.c_str(), sec.size);
    if (_instream.gcount() != sec.size) {
      LOG_ERROR << CYBERTRON_ERROR << RECORD_FILE_READ_ERROR << " file [" << _path << "] read error.";
      return FAIL;
    }
    if (!section->ParseFromString(section_msg)) {
      LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " file [" << _path << "] section ParseFromString error.";
      return FAIL;
    }

    _instream.seekg(pos_old, std::ios::beg);
    return SUCC;
  }

  int64_t _file_size;
  std::ifstream _instream;
  cybertron::proto::HeaderSection _header_section;
  cybertron::proto::IndexSection _index_section;
};

class OutFileOpt : public FileOpt {
 public:
  SMART_PTR_DEFINITIONS(OutFileOpt)
  OutFileOpt();
  virtual ~OutFileOpt();
  int open(const std::string& path) override;
  void close() override;

  const cybertron::proto::HeaderSection& get_header();
  void UpdateMessageNum(std::unordered_map<std::string, int> msg_count);
  void AddChannel(const std::string& channel_name, const std::string& type,
                  const std::string& desc);
  int WriteHeader(const cybertron::proto::HeaderSection& header);
  int WriteHeader();
  void write_header_md5();
  void set_header(const cybertron::proto::HeaderSection& header);
  int WriteChunk(cybertron::proto::ChunkHeader& chunkheader,
                 const cybertron::proto::ChunkSection& chunkbody);
  int WriteReserve(const cybertron::proto::ReserveSection& reserve);
  int WriteParam(const cybertron::proto::ParamSection& param);
  int WriteIndex(const cybertron::proto::IndexSection& index);

  uint64_t get_file_size();
  int get_channel_size();

 private:
  int WriteChunkImpl(cybertron::proto::ChunkHeader& chunkheader,
                     const cybertron::proto::ChunkSection& chunk);
  void _write_header_md5();

  bool _need_write_header = true;
  std::ofstream _outstream;
  cybertron::proto::HeaderSection _header_section;
  cybertron::proto::IndexSection _index_section;
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_RECORDER_FILEOPT_H_
