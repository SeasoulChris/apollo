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

#include <vector>
#include <string>

// #include "fueling/common/record/kinglong/cybertron/common/common.h"
#include "fueling/common/record/kinglong/cybertron/common/macros.h"
#include "fueling/common/record/kinglong/proto/cybertron/record.pb.h"
// #include "parameter_recorder_helper.h"

namespace cybertron {

typedef std::vector<std::string> ChannelVec;
typedef std::unordered_map<std::string, std::string> ChannelTypeMap;

struct RecorderParam {
  RecorderParam()
      : path(""),
        version("1.0.0"),
        compress_type(fueling::common::record::kinglong::proto::cybertron::COMPRESS_NONE),
        chunk_interval(20 * 1000000000L),  // 20s
        chunk_limit(10000000),             // 10M
        segment_interval(0),               // in nanosecond
        begin_time(0),
        end_time(0),
        rate(1.0),
        wait_second(1.0),
        start_second(0) {}

  // common
  std::string path;
  std::vector<std::string> play_paths;
  std::string version;
  // COMPRESS_NONE = 1;
  // COMPRESS_BZ2 = 2;
  fueling::common::record::kinglong::proto::cybertron::CompressType compress_type;
  uint64_t chunk_interval;
  uint64_t chunk_limit;
  // 0:means will not segment
  uint64_t segment_interval;
  ChannelVec channel_vec;
  // in nanosecond
  uint64_t begin_time;
  // in nanosecond
  uint64_t end_time;
  // record
  bool record_all;
  // play
  float rate;
  bool loop_replay;
  float wait_second;
  int64_t start_second;
  // info
  std::vector<std::string> files;
  bool show_freq;
};

}  // namespace cybertron
