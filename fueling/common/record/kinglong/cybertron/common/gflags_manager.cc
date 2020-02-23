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

#include <fstream>
// #include <gflags/gflags.h>

#include "fueling/common/record/kinglong/cybertron/common/gflags_manager.h"
// #include "cybertron/simulator/simulator.h"

namespace cybertron {

DEFINE_string(gflags_load_check, "error", "sometimes gflags load failed.");

GflagsManager::GflagsManager() {}
GflagsManager::~GflagsManager() {}

int GflagsManager::Init() {
  std::lock_guard<std::mutex> lck(_mutex);
  return LoadAll();
}

int GflagsManager::LoadAll() {
  std::vector<std::string> files;
  const std::string run_model = ""; // simulator::Simulation::Instance()->RunModel();

  std::string work_root = WorkRoot();
  std::string gflags_root_path = FileUtil::get_absolute_path(work_root, "conf");

  if (FileUtil::get_file_list(gflags_root_path, ".flag", &files) != SUCC) {
    // LOG_ERROR << CYBERTRON_ERROR << GFLAG_GET_FILE_LIST_ERROR << " gflags_manager_path: "
    //   << gflags_root_path << " get_file_list error.";
    return FAIL;
  }

  std::string module_root = ModuleRoot();
  if (module_root != "./") {
    std::string gflags_module_path =
        FileUtil::get_absolute_path(module_root, "conf");
    if (FileUtil::get_file_list(gflags_module_path, ".flag", &files) != SUCC) {
      // LOG_ERROR << CYBERTRON_ERROR << GFLAG_GET_FILE_LIST_ERROR << " gflags_manager_path : "
      //   << gflags_module_path << " get_file_list error.";
      return FAIL;
    }
  }

  if (run_model == "SIM") {
    FLAGS_flagfile =
        FileUtil::get_absolute_path(gflags_root_path,
            std::string("gflags.") + std::to_string(getpid()) + ".gflags");
  } else {
    FLAGS_flagfile =
        FileUtil::get_absolute_path(gflags_root_path, "gflags.gflags");
  }

  if (FileUtil::Exists(FLAGS_flagfile)) {
    // LOG_WARN << "FLAGS_flagfile: " << FLAGS_flagfile
    //          << " exists. will overwrite it";
  }
  std::ofstream outstream;
  outstream.open(FLAGS_flagfile, std::ios::out);
  if (!outstream || !outstream.is_open()) {
    // LOG_ERROR << CYBERTRON_ERROR << GFLAG_FILE_OPEN_ERROR << " FLAGS_flagfile open error.";
    return FAIL;
  }
  for (auto& file : files) {
    // LOG_INFO << "GFLAGS FILE : " << file;
    std::string content = "# " + file + "\n";
    if (!FileUtil::get_file_content(file, &content)) {
      // LOG_ERROR << CYBERTRON_ERROR << GFLAG_FILE_READ_ERROR << " flagfile read error." << file;
      return FAIL;
    }
    content += "\n\n";
    outstream.write(content.c_str(), content.length());
  }
  outstream.close();
  // LOG_INFO << "GFLAGS FILE MERGED : " << FLAGS_flagfile;
  int fake_argc = 1;
  char** fake_argv = new char* [1]{(char *)"--frame_name=cybertron"};
  google::ParseCommandLineFlags(&fake_argc, &fake_argv, true);

  // LOG_INFO << "finish to load Gflags.";
  delete[] fake_argv;
  _inited = true;

  // if (FLAGS_gflags_load_check == "error") {
  //   LOG_ERROR << "gflags load error";
  //   if (run_model == "SIM") {
  //     simulator::Simulation::Instance()->SetStatus(13);
  //   }
  // }

  return SUCC;
}

int GflagsManager::LoadFile(const std::string& fname) {
  if (fname[0] == '/') {
    FLAGS_flagfile = fname;
  } else {
    std::string work_root = WorkRoot();
    std::string module_root = ModuleRoot();
    std::string file_path;
    if (module_root != "./") {
      file_path = FileUtil::get_absolute_path(module_root, fname);
      if (!FileUtil::Exists(file_path)) {
        file_path = FileUtil::get_absolute_path(work_root, fname);
      }
    }
    FLAGS_flagfile = file_path;
  }
  // LOG_INFO << "GFLAG FILE : " << FLAGS_flagfile;
  int fake_argc = 1;
  char** fake_argv = new char* [1]{(char *)"--frame_name=cybertron"};
  google::ParseCommandLineFlags(&fake_argc, &fake_argv, true);
  delete[] fake_argv;
  return SUCC;
}

}  // namespace cybertron
