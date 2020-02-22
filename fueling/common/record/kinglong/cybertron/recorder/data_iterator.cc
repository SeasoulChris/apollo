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

#include "cybertron/recorder/data_iterator.h"

namespace cybertron {
DataIterator::DataIterator(const DataFile::SharedPtr& data_file) {
  datafile_ = data_file;
  chunk_num_ = datafile_->get_chunk_num();
}

bool DataIterator::Update(iterator* it) {
  cybertron::proto::ChunkSection chunk;
  cybertron::proto::ChunkHeader header;
  while (it->index_ == msgs_.size()) {
    if (chunk_index_ == chunk_num_) {
      return false;
    }

    if (chunk_index_ == 0) {
      datafile_->ReadChunk(chunk_index_, &chunk, &header, false);
    } else {
      datafile_->ReadChunk(&chunk, &header);
    }
    chunk_index_++;
    for (auto& msg : chunk.msgs()) {
      if (!IsValid(msg)) {
        continue;
      }
      msgs_.push_back(std::move(msg));
    }
  }
  return true;
}

bool DataIterator::IsValid(const DataIterator::MessageInstance& msg) {
  for (auto filter : filters_) {
    if (!filter->IsValid(msg)) {
      return false;
    }
  }
  return true;
}

void DataIterator::AddFilter(const BaseFilter::SharedPtr& filter) {
  filters_.push_back(filter);
}

DataIterator::iterator::iterator() {}

DataIterator::iterator::~iterator() {}

DataIterator::iterator::iterator(DataIterator* di, bool end)
    : di_(di), end_(end) {
  if (end_) {
    return;
  }

  if (!di_->Update(this)) {
    end_ = true;
  } else {
    message_instance_ = &di_->msgs_.at(index_);
  }
}

DataIterator::iterator DataIterator::begin() { return iterator(this); }

DataIterator::iterator DataIterator::end() { return iterator(this, true); }

bool DataIterator::iterator::equal(iterator const& other) const {
  if (other.end_) {
    return end_;
  }
  return index_ == other.index_;
}

void DataIterator::iterator::increment() {
  if (++index_ == di_->msgs_.size() && !di_->Update(this)) {
    end_ = true;
    return;
  }
  message_instance_ = &di_->msgs_.at(index_);
}

DataIterator::MessageInstance& DataIterator::iterator::dereference() const {
  return *message_instance_;
}
}
