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

#include <climits>
#include <string>
// #include "cybertron/common/common.h"
#include "fueling/common/record/kinglong/cybertron/common/macros.h"
#include "fueling/common/record/kinglong/proto/cybertron/record.pb.h"

namespace cybertron {
class BaseFilter {
 public:
  SMART_PTR_DEFINITIONS_NOT_COPYABLE(BaseFilter);
  using MessageInstance = fueling::common::record::kinglong::proto::cybertron::SingleMsg;
  virtual bool IsValid(const MessageInstance& msg) const = 0;
};

class TimerFilter : public BaseFilter {
 public:
  TimerFilter(const uint64_t& begin = 0, const uint64_t& end = ULONG_MAX) {
    begin_ = begin;
    end_ = end;
  }

  bool IsValid(const MessageInstance& msg) const override {
    return (msg.time() < end_ && msg.time() > begin_);
  }

 private:
  uint64_t begin_;
  uint64_t end_;
};

/*
class TypeFilter : public BaseFilter {
 public:
  TypeFilter(const std::string& type) {
    type_ = type;
  }

  bool IsValid(const MessageInstance& msg) const override {
    return true;
    //return (msg.type() == type_);
  }

 private:
  std::string type_;
};
*/

class ChannelFilter : public BaseFilter {
 public:
  ChannelFilter(const std::string& channel) { channels_.push_back(channel); }

  ChannelFilter(const std::vector<std::string>& channel) {
    channels_ = channel;
  }

  bool IsValid(const MessageInstance& msg) const override {
    return (std::find(channels_.begin(), channels_.end(), msg.channelname()) !=
            channels_.end());
  }

 private:
  std::vector<std::string> channels_;
};

}  // namespace cybertron
