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

#include "cybertron/common/config_manager.h"

namespace cybertron {

#undef DO_IF
#define DO_IF(condition, code) \
  if (condition) {             \
    code                       \
  }

ConfigManager::ConfigManager() {}

int ConfigManager::Init() {
  std::lock_guard<std::mutex> lck(_mutex);
  return InitInternal();
}

int ConfigManager::InitInternal() {
  RETURN_VAL_IF2(_inited, SUCC);
  GflagsManager::SharedPtr gflags_manager = GflagsManager::Instance();

  DO_IF(gflags_manager == nullptr, {
    LOG_ERROR << CYBERTRON_ERROR << GFLAG_MGR_INSTANCE_ERROR
              << " GflagsManager init error.";
    return FAIL;
  });

  _model_config_map.clear();

  std::vector<std::string> model_config_files;

  std::string work_root = WorkRoot();
  std::string config_root_path = FileUtil::get_absolute_path(work_root, "conf");

  DO_IF(FileUtil::get_file_list(config_root_path, ".cm.config",
                                &model_config_files) != SUCC,
        {
          LOG_ERROR << CYBERTRON_ERROR << CONFIG_GET_FILE_LIST_ERROR
                    << " config_root_path : " << config_root_path
                    << " get_file_list error.";
          return FAIL;
        });

  std::string module_root = ModuleRoot();

  DO_IF(module_root != "./", {
    std::string config_module_path =
        FileUtil::get_absolute_path(module_root, "conf");
    if (FileUtil::get_file_list(config_module_path, ".cm.config",
                                &model_config_files) != SUCC) {
      LOG_ERROR << CYBERTRON_ERROR << CONFIG_GET_FILE_LIST_ERROR
                << " config_root_path : " << config_module_path
                << " get_file_list error.";
      return FAIL;
    }
  });

  for (auto& model_config_file : model_config_files) {
    LOG_INFO << "CONFIG FILE : " << model_config_file;

    std::string config_content;
    DO_IF(!FileUtil::get_file_content(model_config_file, &config_content), {
      LOG_ERROR << CYBERTRON_ERROR << CONFIG_FILE_PARSE_ERROR
                << " failed to get_file_content: " << model_config_file;
      return FAIL;
    });

    MultiModelConfigProto multi_model_config_proto;

    DO_IF(!google::protobuf::TextFormat::ParseFromString(
              config_content, &multi_model_config_proto),
          {
            LOG_ERROR << CYBERTRON_ERROR << CONFIG_FILE_PARSE_ERROR
                      << "invalid MultiModelConfigProto file: "
                      << model_config_file;
            return FAIL;
          });

    for (const ModelConfigProto& model_config_proto :
         multi_model_config_proto.model_configs()) {
      ModelConfig::SharedPtr model_config = ModelConfig::make_shared();
      RETURN_VAL_IF2(model_config == nullptr, FAIL);
      RETURN_VAL_IF2(model_config->reset(model_config_proto) != SUCC, FAIL);

      LOG_INFO << "load ModelConfig succ. name: " << model_config->name();

      std::pair<ModelConfigMapIterator, bool> result =
          _model_config_map.emplace(model_config->name(), model_config);
      DO_IF(!result.second, {
        LOG_WARN << "duplicate ModelConfig, name: " << model_config->name();
        return FAIL;
      });
    }
  }

  LOG_INFO << "finish to load ModelConfigs. num_models: "
           << _model_config_map.size();

  _inited = true;

  return SUCC;
}

int ConfigManager::reset() {
  std::lock_guard<std::mutex> lck(_mutex);
  _inited = false;
  return InitInternal();
}

int ConfigManager::get_model_config(const std::string& model_name,
                                    ModelConfig::SharedPtr* model_config) {
  DO_IF(!_inited && Init() == FAIL, {
    LOG_WARN << "not inited error";
    return FAIL;
  });

  ModelConfigMapConstIterator citer = _model_config_map.find(model_name);

  DO_IF(citer == _model_config_map.end(), {
    LOG_WARN << "not find model[" << model_name << "] error";
    return FAIL;
  });
  *model_config = citer->second;
  return SUCC;
}

ConfigManager::~ConfigManager() { _model_config_map.clear(); }

int ModelConfig::reset(const ModelConfigProto& proto) {
  _name = proto.name();
  _version = proto.version();

  _integer_param_map.clear();
  _uint64_param_map.clear();
  _string_param_map.clear();
  _double_param_map.clear();
  _float_param_map.clear();
  _bool_param_map.clear();
  _array_integer_param_map.clear();
  _array_string_param_map.clear();
  _array_double_param_map.clear();
  _array_float_param_map.clear();
  _array_bool_param_map.clear();

  for (const KeyValueInt& pair : proto.integer_params()) {
    _integer_param_map.emplace(pair.name(), pair.value());
  }

  for (const KeyValueUint64& pair : proto.uint64_params()) {
    _uint64_param_map.emplace(pair.name(), pair.value());
  }

  for (const KeyValueString& pair : proto.string_params()) {
    _string_param_map.emplace(pair.name(), pair.value());
  }

  for (const KeyValueDouble& pair : proto.double_params()) {
    _double_param_map.emplace(pair.name(), pair.value());
  }

  for (const KeyValueFloat& pair : proto.float_params()) {
    _float_param_map.emplace(pair.name(), pair.value());
  }

  for (const KeyValueBool& pair : proto.bool_params()) {
    _bool_param_map.emplace(pair.name(), pair.value());
  }

  for (const KeyValueArrayInt& pair : proto.array_integer_params()) {
    std::vector<int> values;
    RepeatedToVector(pair.values(), &values);
    _array_integer_param_map.emplace(pair.name(), values);
  }

  for (const KeyValueArrayString& pair : proto.array_string_params()) {
    std::vector<std::string> values;
    values.reserve(pair.values_size());
    for (const std::string& value : pair.values()) {
      values.push_back(value);
    }
    _array_string_param_map.emplace(pair.name(), values);
  }

  for (const KeyValueArrayDouble& pair : proto.array_double_params()) {
    std::vector<double> values;
    RepeatedToVector(pair.values(), &values);
    _array_double_param_map.emplace(pair.name(), values);
  }

  for (const KeyValueArrayFloat& pair : proto.array_float_params()) {
    std::vector<float> values;
    RepeatedToVector(pair.values(), &values);
    _array_float_param_map.emplace(pair.name(), values);
  }

  for (const KeyValueArrayBool& pair : proto.array_bool_params()) {
    std::vector<bool> values;
    RepeatedToVector(pair.values(), &values);
    _array_bool_param_map.emplace(pair.name(), values);
  }

  LOG_INFO << "reset ModelConfig. model_name: " << _name
           << " integer_param_map's size: " << _integer_param_map.size()
           << " uint64_param_map's size: " << _uint64_param_map.size()
           << " string_param_map's size: " << _string_param_map.size()
           << " double_param_map's size: " << _double_param_map.size()
           << " float_param_map's size: " << _float_param_map.size()
           << " bool_param_map's size: " << _bool_param_map.size()
           << " array_integer_param_map's size: "
           << _array_integer_param_map.size()
           << " array_string_param_map's size: "
           << _array_string_param_map.size()
           << " array_double_param_map's size: "
           << _array_double_param_map.size()
           << " array_float_param_map's size: " << _array_float_param_map.size()
           << " array_bool_param_map's size: " << _array_bool_param_map.size();

  return SUCC;
}

#undef DO_IF

}  // namespace cybertron
