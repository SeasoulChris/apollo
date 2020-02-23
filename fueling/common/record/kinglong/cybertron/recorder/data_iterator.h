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

#include "fueling/common/record/kinglong/cybertron/recorder/filter.h"
#include "fueling/common/record/kinglong/cybertron/recorder/datafile.h"

namespace cybertron {
class DataIterator {
 public:
  using MessageInstance = fueling::common::record::kinglong::proto::cybertron::SingleMsg;
  class iterator : public boost::iterator_facade<iterator, MessageInstance,
                                                 boost::forward_traversal_tag> {
   public:
    explicit iterator(DataIterator* di, bool end = false);
    iterator();
    virtual ~iterator();

   private:
    friend class DataIterator;
    friend class boost::iterator_core_access;

    bool equal(iterator const& other) const;
    void increment();
    MessageInstance& dereference() const;

    DataIterator* di_;
    bool end_ = false;
    uint64_t index_ = 0;
    MessageInstance* message_instance_;
  };

 public:
  typedef iterator const_iterator;

  DataIterator(const DataFile::SharedPtr& data_file);
  bool Update(iterator* it);
  bool IsValid(const MessageInstance& msg);
  void AddFilter(const BaseFilter::SharedPtr& filter);

  iterator begin();
  iterator end();

 private:
  friend class iterator;
  int chunk_num_ = 0;
  int chunk_index_ = 0;
  DataFile::SharedPtr datafile_ = nullptr;
  std::vector<BaseFilter::SharedPtr> filters_;
  std::vector<MessageInstance> msgs_;
};
}  // namespace cybertron
