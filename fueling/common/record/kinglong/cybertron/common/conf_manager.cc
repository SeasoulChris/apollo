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

#include "boost/regex.hpp"
#include "fueling/common/record/kinglong/cybertron/common/conf_manager.h"
#include "fueling/common/record/kinglong/cybertron/common/gflags_manager.h"

using namespace std;

namespace cybertron {

Config::Config(string filename, string delimiter, string comment)
    : m_Delimiter(delimiter), m_Comment(comment) {
  std::ifstream in(filename.c_str());
  if (!in) {
    // LOG_ERROR << CYBERTRON_ERROR << CONF_FILE_OPEN_ERROR 
    //   << "file[" << filename << "] open error.";
  }
  LoadFile(in, *this);
  in.close();
}

Config::Config() : m_Delimiter(string(1, '=')), m_Comment(string(1, '#')) {}

bool Config::KeyExists(const string& key) const {
  mapci p = m_Contents.find(key);
  return (p != m_Contents.end());
}

void Config::Trim(string& inout_s) {
  static const char whitespace[] = " \n\t\v\r\f";
  inout_s.erase(0, inout_s.find_first_not_of(whitespace));
  inout_s.erase(inout_s.find_last_not_of(whitespace) + 1U);
}

void Config::Remove(const string& key) {
  m_Contents.erase(m_Contents.find(key));
  return;
}

int Config::LoadFile(std::istream& is, Config& cf) {
  typedef string::size_type pos;
  const string& delim = cf.m_Delimiter;  // separator
  const string& comm = cf.m_Comment;     // comment
  const pos skip = delim.length();       // length of separator
  std::string secname = "";

  string nextline = "";  // might need to read ahead to see where value ends
  while (is || nextline.length() > 0) {
    string line;
    if (nextline.length() > 0) {
      line = nextline;  // we read ahead; use it now
      nextline = "";
    } else {
      std::getline(is, line);
    }
    // Ignore comments
    line = line.substr(0, line.find(comm));
    // Parse the line if it contains a delimiter
    pos delimPos = line.find(delim);
    if (delimPos < string::npos) {
      // Extract the key
      string key = line.substr(0, delimPos);
      line.replace(0, delimPos + skip, "");
      // Store key and value
      Config::Trim(key);
      Config::Trim(line);
      std::string newkey = key;
      if (secname != "") {
        newkey = secname + "_" + key;
      }
      auto it = cf.m_Contents.find(newkey);
      if (it != cf.m_Contents.end()) {
        // LOG_ERROR << CYBERTRON_ERROR << CONF_FILE_LOAD_ERROR << " \tconf section[" 
        //   << secname << "] key[" << key << "] exist conflict error.";
        return FAIL;
      }
      cf.m_Contents[newkey] = line;  // overwrites if key is repeated
      // LOG_INFO << "\tconf section[" << secname << "] key[" << key << "] value["
      //          << line << "]";
    } else {
      boost::cmatch mat;
      try {
        boost::regex reg("\\[(.*)\\]");
        if (boost::regex_match(line.c_str(), mat, reg)) {
          for (boost::cmatch::iterator itr = mat.begin(); itr != mat.end();
               ++itr) {
            secname = *itr;
          }
        }
      } catch (boost::exception& e) {
        // LOG_WARN << "Invalid regular expression";
      }
    }
  }
  return SUCC;
}

bool Config::FileExist(std::string filename) {
  bool exist = false;
  std::ifstream in(filename.c_str());
  if (in) exist = true;
  return exist;
}

int Config::ReadFile(string filename, string delimiter, string comment) {
  m_Delimiter = delimiter;
  m_Comment = comment;
  std::ifstream in(filename.c_str());
  if (!in) {
    // LOG_ERROR << CYBERTRON_ERROR << CONF_FILE_OPEN_ERROR << " file[" << filename << "] open error.";
    return FAIL;
  }
  int ret = LoadFile(in, *this);
  in.close();
  return ret;
}

ConfManager::ConfManager() {}
ConfManager::~ConfManager() {}

int ConfManager::Init() {
  std::lock_guard<std::mutex> lck(_mutex);
  conf_ = Config::make_shared();
  if (conf_ == nullptr) {
    // LOG_ERROR << CYBERTRON_ERROR << NEW_MEMORY_ERROR << " Config::make_shared() error.";
    return FAIL;
  }
  return InitInternal();
}

int ConfManager::InitInternal() {
  if (_inited) {
    return SUCC;
  }

  std::vector<std::string> files;

  std::string work_root = WorkRoot();
  std::string conf_root_path = FileUtil::get_absolute_path(work_root, "conf");
  if (FileUtil::get_file_list(conf_root_path, ".cy.config", &files) != SUCC) {
    // LOG_DEBUG << "conf_manager_path: " << conf_root_path
    //           << " get_file_list error.";
    return FAIL;
  }

  std::string module_root = ModuleRoot();
  if (module_root != "./") {
    std::string conf_module_path =
        FileUtil::get_absolute_path(module_root, "conf");
    if (FileUtil::get_file_list(conf_module_path, ".cy.config", &files) !=
        SUCC) {
      // LOG_ERROR << CYBERTRON_ERROR << CONF_GET_FILE_LIST_ERROR << " conf_manager_path: "
      //   << conf_module_path << " get_file_list error.";
      return FAIL;
    }
  }

  for (auto& file : files) {
    // LOG_INFO << "CONF FILE : " << file;
    if (conf_->ReadFile(file) != SUCC) {
      // LOG_ERROR << CYBERTRON_ERROR <<  << "conf ReadFile[" << file << "] error.";
      return FAIL;
    }
  }
  // LOG_INFO << "finish to load Conf.";
  _inited = true;
  return SUCC;
}

}  // namespace cybertron
