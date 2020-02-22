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

#include "cybertron/common/registerer.h"
#include "cybertron/common/error_code.h"

namespace cybertron {

BaseClassMap& global_factory_map() {
  static BaseClassMap factory_map;
  return factory_map;
}

bool get_registered_classes(
    const std::string& base_class_name,
    std::vector<std::string>* registered_derived_classes_names) {
  CHECK_NOTNULL(registered_derived_classes_names);
  BaseClassMap& map = global_factory_map();
  auto iter = map.find(base_class_name);
  if (iter == map.end()) {
    LOG_ERROR << CYBERTRON_ERROR << CLASS_REGISTER_ERROR << " class not registered:" << base_class_name;
    return false;
  }
  for (auto pair : iter->second) {
    registered_derived_classes_names->push_back(pair.first);
  }
  return true;
}

}  // namespace cybertron
