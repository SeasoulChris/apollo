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

#include "fueling/common/record/kinglong/cybertron/recorder/chunk.h"

#include "fueling/common/record/kinglong/cybertron/common/define.h"

namespace cybertron {

using fueling::common::record::kinglong::proto::cybertron::SingleMsg;

Chunk::Chunk() { reset(); }

Chunk::~Chunk() {}

int Chunk::write(const SingleMsg& msg) {
  std::lock_guard<std::mutex> lck(mutex_);
  if (chunk_section_.msgs_size() == 0) {
    SingleMsg* singlemsg = chunk_section_.add_msgs();
    *singlemsg = msg;
    chunk_header_.set_begintime(msg.time());
    chunk_header_.set_endtime(msg.time());
    chunk_header_.set_msgnum(1);
    chunk_header_.set_rawsize(msg.msg().size());
  } else {
    SingleMsg* singlemsg = chunk_section_.add_msgs();
    *singlemsg = msg;
    if (msg.time() < chunk_header_.begintime()) {
      chunk_header_.set_begintime(msg.time());
    }
    chunk_header_.set_endtime(msg.time());
    chunk_header_.set_msgnum(chunk_header_.msgnum() + 1);
    chunk_header_.set_rawsize(chunk_header_.rawsize() + msg.msg().size());
  }
  return SUCC;
}

void Chunk::reset() {
  chunk_section_.clear_msgs();
  chunk_header_.set_begintime(0);
  chunk_header_.set_endtime(0);
  chunk_header_.set_msgnum(0);
  chunk_header_.set_rawsize(0);
}

uint64_t Chunk::BeginTime() const { return chunk_header_.begintime(); }

uint64_t Chunk::RawSize() const { return chunk_header_.rawsize(); }
}
