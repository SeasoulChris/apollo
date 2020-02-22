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

#ifndef INCLUDE_CYBERTRON_COMMON_HEADER_WRAPPER_H_
#define INCLUDE_CYBERTRON_COMMON_HEADER_WRAPPER_H_

#include "cybertron/common/common.h"
//#include "cybertron/dag_streaming/receiver.h"
//#include "cybertron/dag_streaming/sender.h"
//#include "cybertron/proto/dag_config.pb.h"

namespace cybertron {

template <typename M, typename Enable = void>
class HeaderWrapper {
 public:
  static const cybertron::proto::CyberHeader* get_header(
      const std::shared_ptr<const M>& msg) {
    return nullptr;
  }
  static void set_header(const cybertron::proto::CyberHeader& header,
                         std::shared_ptr<M>& msg, const std::string& name) {
    return;
  }
  static void ShowTrace(const std::shared_ptr<const M>& msg,
                        const std::string& mark) {
    return;
  }
};

template <typename M>
class HeaderWrapper<
    M, typename std::enable_if<
           std::is_member_function_pointer<decltype(&M::cyber_header)>::value &&
           std::is_member_function_pointer<decltype(
               &M::has_cyber_header)>::value>::type> {
 public:
  static const cybertron::proto::CyberHeader* get_header(
      const std::shared_ptr<const M>& msg) {
    if (!msg->has_cyber_header()) {
      return nullptr;
    }

    if (msg->cyber_header().meta_stamp() != 0) {
      M* mutable_msg = const_cast<M*>(msg.get());
      return mutable_msg->mutable_cyber_header();
    }
    return nullptr;
  }

  static void set_header(const cybertron::proto::CyberHeader& header,
                         std::shared_ptr<M>& msg, const std::string& name) {
    if (msg->mutable_cyber_header()->meta_stamp() == 0 &&
        header.meta_stamp() > 0) {
      msg->mutable_cyber_header()->set_meta_stamp(header.meta_stamp());
    }
    msg->mutable_cyber_header()->set_stamp(
        cybertron::Time::Now().ToNanosecond());
    return;
  }
  static void ShowTrace(const std::shared_ptr<const M>& msg,
                        const std::string& mark) {
    if (!msg->has_cyber_header()) {
      return;
    }
    return;
  }
};

template <typename M, typename Enable = void>
class CommonHeaderWrapper {
 public:
  static const adu::common::header::Header* get_header(
      const std::shared_ptr<const M>& msg) {
    return nullptr;
  }
};

template <typename M>
class CommonHeaderWrapper<
    M, typename std::enable_if<
           std::is_member_function_pointer<decltype(&M::header)>::value &&
           !std::is_member_function_pointer<decltype(
               &M::has_header)>::value>::type> {
 public:
  static const adu::common::header::Header* get_header(
      const std::shared_ptr<const M>& msg) {
    return nullptr;
  }
};

template <typename M>
class CommonHeaderWrapper<
    M, typename std::enable_if<
           std::is_member_function_pointer<decltype(&M::header)>::value &&
           std::is_member_function_pointer<decltype(
               &M::has_header)>::value>::type> {
 public:
  static const adu::common::header::Header* get_header(
      const std::shared_ptr<const M>& msg) {
    if (!msg->has_header()) {
      return nullptr;
    }
    return &(msg->header());
  }
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_COMMON_HEADER_WRAPPER_H_
