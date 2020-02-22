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

#ifndef INCLUDE_CYBERTRON_COMMON_TYPES_H_
#define INCLUDE_CYBERTRON_COMMON_TYPES_H_

#include <google/protobuf/descriptor.h>
#include "cybertron/common/message_base.h"

namespace cybertron {

// Template needs
struct NullType : public MessageBase {
  static const ::google::protobuf::Descriptor* descriptor() { return NULL; }
  bool ParseFromString(const std::string& str) {
    (void)str;
    return true;
  }
  int seq() { return 0; }
};
typedef std::shared_ptr<NullType const> NullTypeConstPtr;
template <class M>
struct NullFilter {};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_COMMON_TYPES_H_
