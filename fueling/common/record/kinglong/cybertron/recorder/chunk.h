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

#ifndef INCLUDE_CYBERTRON_RECORDER_CHUNK_H_
#define INCLUDE_CYBERTRON_RECORDER_CHUNK_H_

#include <mutex>
#include <unordered_map>
#include "cybertron/recorder/param.h"

namespace cybertron {

class Chunk {
 public:
  SMART_PTR_DEFINITIONS(Chunk)
  Chunk();
  virtual ~Chunk();

  void reset();
  int write(const cybertron::proto::SingleMsg& msg);

  uint64_t BeginTime() const;
  uint64_t RawSize() const;

 private:
  friend class DataFile;
  std::mutex mutex_;
  cybertron::proto::ChunkSection chunk_section_;
  cybertron::proto::ChunkHeader chunk_header_;
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_RECORDER_CHUNK_H_
