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

#ifndef INCLUDE_CYBERTRON_COMMON_GFLAGS_MANAGER_H_
#define INCLUDE_CYBERTRON_COMMON_GFLAGS_MANAGER_H_

#include <mutex>
#include <gflags/gflags.h>

#include "cybertron/common/define.h"
#include "cybertron/common/macros.h"
#include "cybertron/common/logger.h"
#include "cybertron/common/file_util.h"
#include "cybertron/common/environment.h"
#include "cybertron/common/error_code.h"

DECLARE_string(flagfile);

namespace cybertron {

class GflagsManager {
 public:
  SMART_PTR_DEFINITIONS(GflagsManager)
  virtual ~GflagsManager();
  int LoadAll();
  int LoadFile(const std::string& fname);

 private:
  bool _inited = false;
  std::mutex _mutex;  // multi-thread init safe.

  DECLARE_SINGLETON_WITH_INIT(GflagsManager);
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_COMMON_GFLAGS_MANAGER_H_
