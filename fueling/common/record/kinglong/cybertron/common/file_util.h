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

#include <unistd.h>

#include <string>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "fueling/common/record/kinglong/cybertron/common/macros.h"
// #include "cybertron/common/logger.h"

namespace cybertron {

enum FileType { TYPE_FILE, TYPE_DIR };

// file name compared type
enum FileCompareType {
  FCT_DIGITAL = 0,
  FCT_LEXICOGRAPHICAL = 1,
  FCT_UNKNOWN = 8
};

class FileUtil {
 public:
  FileUtil() {}

  ~FileUtil() {}

  static std::string Pwd();

  static bool Exists(const std::string& filename);

  // check whether file exists with [suffix] extension in [path]
  static bool Exists(const std::string& path, const std::string& suffix);

  static bool get_type(const std::string& filename, FileType* type);

  static bool DeleteFile(const std::string& filename);

  static bool RenameFile(const std::string& old_file,
                         const std::string& new_file);

  static bool CreateDir(const std::string& dir);

  static bool get_file_content(const std::string& path, std::string* content);
  static bool ReadLines(const std::string& path,
                        std::vector<std::string>* lines);

  static std::string RemoveFileSuffix(std::string filename);

  // TODO: this function stay just for compatibility,
  //                     should be removed later
  static int get_file_list(std::vector<std::string>& files,
                           const std::string path,
                           const std::string suffix = "");

  static int get_file_list(const std::string& path,
                           std::vector<std::string>* files);

  static int get_file_list(const std::string& path, const std::string& suffix,
                           std::vector<std::string>* files);

  static std::string get_absolute_path(const std::string& prefix,
                                       const std::string& relative_path);

  // get file name
  // "/home/work/data/1.txt" -> 1
  static void get_file_name(const std::string& file, std::string* name);

  // return -1 when error occurred
  static int NumLines(const std::string& filename);

  // compare two file's name by digital value
  // "/home/work/data/1.txt" < "/home/user/data/10.txt"
  // "1.txt" < "./data/2.txt"
  static bool CompareFileByDigital(const std::string& file_left,
                                   const std::string& file_right);

  // compare two file's name by lexicographical order
  static bool CompareFileByLexicographical(const std::string& file_left,
                                           const std::string& file_right);

 private:
  static bool CompareFile(const std::string& file_left,
                          const std::string& file_right, FileCompareType type);

  DISALLOW_COPY_AND_ASSIGN(FileUtil);
};

}  // namespace cybertron
