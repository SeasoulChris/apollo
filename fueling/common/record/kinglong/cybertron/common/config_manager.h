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

#ifndef INCLUDE_CYBERTRON_COMMON_CONFIG_MANAGER_H_
#define INCLUDE_CYBERTRON_COMMON_CONFIG_MANAGER_H_

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <typeinfo>
#include <stdlib.h>

#include <google/protobuf/message.h>

#include "cybertron/common/define.h"
#include "cybertron/common/macros.h"
#include "cybertron/common/logger.h"
#include "cybertron/common/file_util.h"
#include "cybertron/common/gflags_manager.h"
#include "cybertron/common/conf_manager.h"
#include "cybertron/common/message_base.h"
#include "cybertron/common/environment.h"
#include "cybertron/common/error_code.h"

namespace cybertron {

class ModelConfigProto;

class ModelConfig {
 public:
  SMART_PTR_DEFINITIONS(ModelConfig)
  ModelConfig() {}
  ~ModelConfig() {}

  int reset(const ModelConfigProto& proto);

  std::string name() const { return _name; }

  int get_value(const std::string& name, int* value) const {
    return get_value_from_map<int>(name, _integer_param_map, value);
  }

  int get_value(const std::string& name, std::string* value) const {
    return get_value_from_map<std::string>(name, _string_param_map, value);
  }

  int get_value(const std::string& name, double* value) const {
    return get_value_from_map<double>(name, _double_param_map, value);
  }

  int get_value(const std::string& name, float* value) const {
    return get_value_from_map<float>(name, _float_param_map, value);
  }

  int get_value(const std::string& name, bool* value) const {
    return get_value_from_map<bool>(name, _bool_param_map, value);
  }

  int get_value(const std::string& name, uint64_t* value) const {
    return get_value_from_map<uint64_t>(name, _uint64_param_map, value);
  }

  int get_value(const std::string& name, std::vector<int>* values) const {
    return get_value_from_map<std::vector<int> >(name, _array_integer_param_map,
                                                 values);
  }

  int get_value(const std::string& name, std::vector<double>* values) const {
    return get_value_from_map<std::vector<double> >(
        name, _array_double_param_map, values);
  }

  int get_value(const std::string& name, std::vector<float>* values) const {
    return get_value_from_map<std::vector<float> >(name, _array_float_param_map,
                                                   values);
  }

  int get_value(const std::string& name,
                std::vector<std::string>* values) const {
    return get_value_from_map<std::vector<std::string> >(
        name, _array_string_param_map, values);
  }

  int get_value(const std::string& name, std::vector<bool>* values) const {
    return get_value_from_map<std::vector<bool> >(name, _array_bool_param_map,
                                                  values);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(ModelConfig);

  template <typename T>
  int get_value_from_map(const std::string& name,
                         const std::map<std::string, T>& container,
                         T* value) const;

  template <typename T>
  void RepeatedToVector(
      const google::protobuf::RepeatedField<T>& repeated_values,
      std::vector<T>* vec_values);

  std::string _name;
  std::string _version;

  typedef std::map<std::string, int> IntegerParamMap;
  typedef std::map<std::string, std::string> StringParamMap;
  typedef std::map<std::string, double> DoubleParamMap;
  typedef std::map<std::string, float> FloatParamMap;
  typedef std::map<std::string, bool> BoolParamMap;
  typedef std::map<std::string, uint64_t> Uint64_tParamMap;
  typedef std::map<std::string, std::vector<int> > ArrayIntegerParamMap;
  typedef std::map<std::string, std::vector<std::string> > ArrayStringParamMap;
  typedef std::map<std::string, std::vector<double> > ArrayDoubleParamMap;
  typedef std::map<std::string, std::vector<float> > ArrayFloatParamMap;
  typedef std::map<std::string, std::vector<bool> > ArrayBoolParamMap;

  IntegerParamMap _integer_param_map;
  StringParamMap _string_param_map;
  DoubleParamMap _double_param_map;
  FloatParamMap _float_param_map;
  BoolParamMap _bool_param_map;
  Uint64_tParamMap _uint64_param_map;
  ArrayIntegerParamMap _array_integer_param_map;
  ArrayStringParamMap _array_string_param_map;
  ArrayDoubleParamMap _array_double_param_map;
  ArrayFloatParamMap _array_float_param_map;
  ArrayBoolParamMap _array_bool_param_map;
};

class ConfigManager {
 public:
  SMART_PTR_DEFINITIONS(ConfigManager)
  virtual ~ConfigManager();
  int reset();
  int get_model_config(const std::string& model_name,
                       ModelConfig::SharedPtr* model_config);
  size_t num_models() const { return _model_config_map.size(); }
  const std::string& work_root() const { return _work_root; }
  const std::string& adu_data() const { return _adu_data; }
  void set_adu_data(const std::string& adu_data) { _adu_data = adu_data; }
  void set_work_root(const std::string& work_root) { _work_root = work_root; }

 private:
  int InitInternal();

  typedef std::map<std::string, ModelConfig::SharedPtr> ModelConfigMap;
  typedef ModelConfigMap::iterator ModelConfigMapIterator;
  typedef ModelConfigMap::const_iterator ModelConfigMapConstIterator;

  // key: model_name
  ModelConfigMap _model_config_map;
  std::mutex _mutex;  // multi-thread init safe.
  bool _inited = false;
  std::string _work_root;  // ConfigManager work root dir.
  std::string _adu_data;

  DECLARE_SINGLETON_WITH_INIT(ConfigManager);
};

template <typename T>
int ModelConfig::get_value_from_map(const std::string& name,
                                    const std::map<std::string, T>& container,
                                    T* value) const {
  typename std::map<std::string, T>::const_iterator citer =
      container.find(name);

  if (citer == container.end()) {
    return FAIL;
  }

  *value = citer->second;
  return SUCC;
}

template <typename T>
void ModelConfig::RepeatedToVector(
    const google::protobuf::RepeatedField<T>& repeated_values,
    std::vector<T>* vec_list) {
  vec_list->reserve(repeated_values.size());
  for (T value : repeated_values) {
    vec_list->push_back(value);
  }
}

class ConfigManagerError {
 public:
  ConfigManagerError(const std::string& error_info) : _error_info(error_info) {}
  std::string what() const { return _error_info; }

 private:
  std::string _error_info;
};

template <typename T>
class ConfigRead {
 public:
  static T read(const ModelConfig& config, const std::string& name) {
    T ret;
    if (!config.get_value(name, &ret)) {
      std::stringstream ss;
      ss << "Config name:" << config.name() << " read failed. "
         << "type:" << typeid(T).name() << " name:" << name;
      throw ConfigManagerError(ss.str());
    }
    return ret;
  }
};

template <typename T>
class ConfigRead<std::vector<T> > {
 public:
  static std::vector<T> read(const ModelConfig& config,
                             const std::string& name) {
    std::vector<T> ret;
    if (!config.get_value(name, &ret)) {
      std::stringstream ss;
      ss << "Config name:" << config.name() << " read failed. "
         << "type:vector<" << typeid(T).name() << "> name:" << name;
      throw ConfigManagerError(ss.str());
    }
    return std::move(ret);
  }
};

}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_COMMON_CONFIG_MANAGER_H_
