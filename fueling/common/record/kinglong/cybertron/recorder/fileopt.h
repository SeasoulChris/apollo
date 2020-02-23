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
#include <fstream>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

#include "fueling/common/record/kinglong/cybertron/common/define.h"
#include "fueling/common/record/kinglong/cybertron/common/macros.h"
#include "fueling/common/record/kinglong/cybertron/common/md5.h"
#include "fueling/common/record/kinglong/proto/cybertron/record.pb.h"
#include "fueling/common/record/kinglong/cybertron/recorder/compress.h"

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
  Section(const fueling::common::record::kinglong::proto::cybertron::SectionType& stype);
  int64_t size;
  fueling::common::record::kinglong::proto::cybertron::SectionType type;
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
  const fueling::common::record::kinglong::proto::cybertron::HeaderSection& get_header();
  fueling::common::record::kinglong::proto::cybertron::IndexSection get_index();
  int ReadChunk(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody);
  int ReadIndex(fueling::common::record::kinglong::proto::cybertron::IndexSection* index);
  // int ReadChunk(int chunk_idx, fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody);
  int ResetToChunk();
  int ReadChunk(int chunk_idx, fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody, bool reset = true);
  int ReadReserve(fueling::common::record::kinglong::proto::cybertron::ReserveSection* reserve);
  int ReadParam(fueling::common::record::kinglong::proto::cybertron::ParamSection* param);
  int ReadIndex();
  int ReadHeader();
  bool IsEnd();

 private:
  int FormatCheck(const std::string& path);
  int ReadHeaderImpl(fueling::common::record::kinglong::proto::cybertron::HeaderSection* header);
  int ReadChunkImplByIndex(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                    fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody);
  int ReadChunkImplBySearch(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                    fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody);
  int ReadChunkHeaderImplByIndex(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader);
  int ReadChunkHeaderImplBySearch(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader);
  int ReadChunkBodyImpl(fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                        fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody);
  int ReadChunkIndexImpl(uint32_t chunk_idx,
                         fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody);
  int ReadChunkIndexImpl(uint32_t chunk_idx,
                         fueling::common::record::kinglong::proto::cybertron::ChunkHeader* chunkheader,
                         fueling::common::record::kinglong::proto::cybertron::ChunkSection* chunkbody);
  int ReadIndexImpl(fueling::common::record::kinglong::proto::cybertron::IndexSection* index, bool flag = false);
  template <typename T>
  int _read_section(T* section,
                    const fueling::common::record::kinglong::proto::cybertron::SectionType& sectiontype) {
    if (!_instream || !_instream.is_open()) {
      // LOG_ERROR << CYBERTRON_ERROR << FILE_NOT_OPEN_ERROR << " file [" << _path << "] not open error.";
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
      // LOG_INFO << "NOTE: there is no ["
      //           << fueling::common::record::kinglong::proto::cybertron::SectionType_Name(sectiontype)
      //           << "] .";
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
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_FILE_READ_ERROR << " file [" << _path << "] read error.";
      return FAIL;
    }
    std::string section_msg;
    section_msg.resize(sec.size);
    _instream.read((char*)section_msg.c_str(), sec.size);
    if (_instream.gcount() != sec.size) {
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_FILE_READ_ERROR << " file [" << _path << "] read error.";
      return FAIL;
    }
    if (!section->ParseFromString(section_msg)) {
      // LOG_ERROR << CYBERTRON_ERROR << RECORD_PARSE_STR_ERROR << " file [" << _path << "] section ParseFromString error.";
      return FAIL;
    }

    _instream.seekg(pos_old, std::ios::beg);
    return SUCC;
  }

  int64_t _file_size;
  std::ifstream _instream;
  fueling::common::record::kinglong::proto::cybertron::HeaderSection _header_section;
  fueling::common::record::kinglong::proto::cybertron::IndexSection _index_section;
};

class OutFileOpt : public FileOpt {
 public:
  SMART_PTR_DEFINITIONS(OutFileOpt)
  OutFileOpt();
  virtual ~OutFileOpt();
  int open(const std::string& path) override;
  void close() override;

  const fueling::common::record::kinglong::proto::cybertron::HeaderSection& get_header();
  void UpdateMessageNum(std::unordered_map<std::string, int> msg_count);
  void AddChannel(const std::string& channel_name, const std::string& type,
                  const std::string& desc);
  int WriteHeader(const fueling::common::record::kinglong::proto::cybertron::HeaderSection& header);
  int WriteHeader();
  void write_header_md5();
  void set_header(const fueling::common::record::kinglong::proto::cybertron::HeaderSection& header);
  int WriteChunk(fueling::common::record::kinglong::proto::cybertron::ChunkHeader& chunkheader,
                 const fueling::common::record::kinglong::proto::cybertron::ChunkSection& chunkbody);
  int WriteReserve(const fueling::common::record::kinglong::proto::cybertron::ReserveSection& reserve);
  int WriteParam(const fueling::common::record::kinglong::proto::cybertron::ParamSection& param);
  int WriteIndex(const fueling::common::record::kinglong::proto::cybertron::IndexSection& index);

  uint64_t get_file_size();
  int get_channel_size();

 private:
  int WriteChunkImpl(fueling::common::record::kinglong::proto::cybertron::ChunkHeader& chunkheader,
                     const fueling::common::record::kinglong::proto::cybertron::ChunkSection& chunk);
  void _write_header_md5();

  bool _need_write_header = true;
  std::ofstream _outstream;
  fueling::common::record::kinglong::proto::cybertron::HeaderSection _header_section;
  fueling::common::record::kinglong::proto::cybertron::IndexSection _index_section;
};

}  // namespace cybertron
