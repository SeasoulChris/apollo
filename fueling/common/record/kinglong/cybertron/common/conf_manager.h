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

#include <mutex>
#include <iostream>
#include <fstream>

// #include "fueling/common/record/kinglong/cybertron/common/logger.h"
#include "fueling/common/record/kinglong/cybertron/common/define.h"
#include "fueling/common/record/kinglong/cybertron/common/file_util.h"
#include "fueling/common/record/kinglong/cybertron/common/environment.h"
#include "fueling/common/record/kinglong/cybertron/common/error_code.h"

namespace cybertron {

#ifndef READ_CONF
#define READ_CONF(secname, key, value)                             \
  do {                                                             \
    cybertron::ConfManager::SharedPtr confmgr =                    \
        cybertron::ConfManager::Instance();                        \
    if (confmgr == nullptr) {                                      \
      return cybertron::FAIL;                                      \
    }                                                              \
    if (confmgr->Read(secname, key, &value) != cybertron::SUCC) {  \
      return cybertron::FAIL;                                      \
    }                                                              \
  } while (0)
#endif

#ifndef READ_CONF_WITH_DEFAULT
#define READ_CONF_WITH_DEFAULT(secname, key, value, default_value) \
  do {                                                             \
    value = default_value;                                         \
    cybertron::ConfManager::SharedPtr confmgr =                    \
        cybertron::ConfManager::Instance();                        \
    if (confmgr != nullptr) {                                      \
      if (confmgr->Read(secname, key, &value) != cybertron::SUCC) {           \
      }                                                            \
    }                                                              \
  } while (0)
#endif

class Config {
 public:
  SMART_PTR_DEFINITIONS(Config)
  Config(std::string filename, std::string delimiter = "=",
         std::string comment = "#");
  Config();
  template <class T>
  int Read(const std::string& in_key, T* in_value) const;
  template <class T>
  int Read(const std::string& secname, const std::string& in_key,
           T* in_value) const;
  bool FileExist(std::string filename);
  int ReadFile(std::string filename, std::string delimiter = "=",
               std::string comment = "#");
  int LoadFile(std::istream& is, Config& cf);

  // Check whether key exists in configuration
  bool KeyExists(const std::string& in_key) const;

  // Modify keys and values
  template <class T>
  void Add(const std::string& in_key, const T& in_value);
  void Remove(const std::string& in_key);

  // Check or change configuration syntax
  std::string GetDelimiter() const { return m_Delimiter; }
  std::string GetComment() const { return m_Comment; }
  std::string SetDelimiter(const std::string& in_s) {
    std::string old = m_Delimiter;
    m_Delimiter = in_s;
    return old;
  }
  std::string SetComment(const std::string& in_s) {
    std::string old = m_Comment;
    m_Comment = in_s;
    return old;
  }

 private:
  template <class T>
  static std::string T_as_string(const T& t);
  template <class T>
  static T string_as_T(const std::string& s);
  static void Trim(std::string& inout_s);

  std::string m_Delimiter;  //!< separator between key and value
  std::string m_Comment;    //!< separator between value and comments
  std::map<std::string, std::string> m_Contents;  //!< extracted keys and values

  typedef std::map<std::string, std::string>::iterator mapi;
  typedef std::map<std::string, std::string>::const_iterator mapci;
};

class ConfManager {
 public:
  SMART_PTR_DEFINITIONS(ConfManager)
  virtual ~ConfManager();
  template <class T>
  int Read(const std::string& secname, const std::string& key, T* value) const;

 private:
  int InitInternal();
  bool _inited = false;
  std::mutex _mutex;  // multi-thread init safe.
  Config::SharedPtr conf_;

  DECLARE_SINGLETON_WITH_INIT(ConfManager);
};

/* static */
template <class T>
std::string Config::T_as_string(const T& t) {
  // Convert from a T to a string
  // Type T must support << operator
  std::ostringstream ost;
  ost << t;
  return ost.str();
}

template <class T>
T Config::string_as_T(const std::string& s) {
  // Convert from a string to a T
  // Type T must support >> operator
  T t;
  std::istringstream ist(s);
  ist >> t;
  return t;
}

template <>
inline std::string Config::string_as_T<std::string>(const std::string& s) {
  return s;
}

template <>
inline int Config::string_as_T<int>(const std::string& s) {
  return std::atoi(s.c_str());
}

template <>
inline long Config::string_as_T<long>(const std::string& s) {
  return std::atol(s.c_str());
}

template <>
inline long long Config::string_as_T<long long>(const std::string& s) {
  return std::atoll(s.c_str());
}

template <>
inline double Config::string_as_T<double>(const std::string& s) {
  return std::atof(s.c_str());
}

template <>
inline float Config::string_as_T<float>(const std::string& s) {
  return std::atof(s.c_str());
}

template <>
inline bool Config::string_as_T<bool>(const std::string& s) {
  bool b = true;
  std::string sup = s;
  for (std::string::iterator p = sup.begin(); p != sup.end(); ++p)
    *p = toupper(*p);  // make string all caps
  if (sup == std::string("FALSE") || sup == std::string("F") ||
      sup == std::string("NO") || sup == std::string("N") ||
      sup == std::string("0") || sup == std::string("NONE"))
    b = false;
  return b;
}

template <class T>
int Config::Read(const std::string& key, T* value) const {
  mapci p = m_Contents.find(key);
  if (p == m_Contents.end()) {
    return FAIL;
  }
  (*value) = string_as_T<T>(p->second);
  return SUCC;
}

template <class T>
int Config::Read(const std::string& secname, const std::string& key,
                 T* value) const {
  mapci p = m_Contents.find(secname + "_" + key);
  if (p == m_Contents.end()) {
    return FAIL;
  }
  (*value) = string_as_T<T>(p->second);
  return SUCC;
}

template <class T>
int ConfManager::Read(const std::string& secname, const std::string& key,
                      T* value) const {
  if (conf_ == nullptr) {
    // LOG_ERROR << CYBERTRON_ERROR << CONF_MGR_INSTANCE_ERROR << "conf_ be null error.";
    return FAIL;
  }
  return conf_->Read(secname, key, value);
}

}  // namespace cybertron
