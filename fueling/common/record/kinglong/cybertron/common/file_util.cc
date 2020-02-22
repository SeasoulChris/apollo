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

// #include "cybertron/common/common.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstring>
#include <fstream>

#include "fueling/common/record/kinglong/cybertron/common/file_util.h"
#include "fueling/common/record/kinglong/cybertron/common/error_code.h"
#include "fueling/common/record/kinglong/cybertron/common/define.h"

#if !defined(_RETURN_VAL_IF2__)
#define _RETURN_VAL_IF2__
#define RETURN_VAL_IF2(condition, val) \
  if (condition) {                     \
    return (val);                      \
  }
#endif

namespace cybertron {

using std::count;
using std::istreambuf_iterator;
using std::string;
using std::vector;

std::string FileUtil::Pwd() {
  static int length = 1024;
  char buff[length];
  getcwd(buff, length);
  return std::string(buff);
}

bool FileUtil::get_type(const string& filename, FileType* type) {
  struct stat stat_buf;
  if (lstat(filename.c_str(), &stat_buf) != 0) {
    return false;
  }
  if (S_ISDIR(stat_buf.st_mode) != 0) {
    *type = TYPE_DIR;
  } else if (S_ISREG(stat_buf.st_mode) != 0) {
    *type = TYPE_FILE;
  } else {
    // LOG_WARN << "failed to get type: " << filename;
    return false;
  }
  return true;
}

bool FileUtil::DeleteFile(const string& filename) {
  if (!Exists(filename)) {
    return true;
  }
  FileType type;
  if(!get_type(filename, &type)) {
    return false;
  }
  // delete files directly
  if (type == TYPE_FILE) {
    if (remove(filename.c_str()) != 0) {
      return false;
    }
    return true;
  }

  // delete iteratively if it's a directory
  DIR* dir = opendir(filename.c_str());
  if (dir == NULL) {
    return false;
  }
  dirent* dir_info = NULL;
  while ((dir_info = readdir(dir)) != NULL) {
    if (strcmp(dir_info->d_name, ".") == 0 ||
        strcmp(dir_info->d_name, "..") == 0) {
      continue;
    }
    // concatenate path
    string temp_file = filename + "/" + string(dir_info->d_name);
    FileType temp_type;
    if (!get_type(temp_file, &temp_type)) {
      // LOG_WARN << "failed to get file type: " << temp_file;
      closedir(dir);
      return false;
    }

    if (type == TYPE_DIR && !DeleteFile(temp_file)) {
        return false;
    }

    if (remove(temp_file.c_str()) != 0) {
      // LOG_WARN << "failed to remove file: " << temp_file;
      return false;
    }
  }
  closedir(dir);
  RETURN_VAL_IF2(remove(filename.c_str()) != 0, false);
  return true;
}

bool FileUtil::Exists(const string& file) {
  int ret = access(file.c_str(), F_OK);
  if (ret != 0) {
    // LOG_INFO << "file not exist. file: " << file << " ret: " <<
    // strerror(errno);
    return false;
  }
  return true;
}

bool FileUtil::Exists(const string& path, const string& suffix) {
  boost::filesystem::recursive_directory_iterator itr(path);
  while (itr != boost::filesystem::recursive_directory_iterator()) {
    if (boost::algorithm::ends_with(itr->path().string(), suffix)) {
      return true;
    }
    ++itr;
  }
  return false;
}

bool FileUtil::RenameFile(const string& old_file, const string& new_file) {
  // delete it first for generality
  RETURN_VAL_IF2(!DeleteFile(new_file), false);
  int ret = rename(old_file.c_str(), new_file.c_str());
  if (ret != 0) {
    // LOG_WARN << "failed to rename [old file: " << old_file
    //         << "] to [newfile: " << new_file << "] [err: " << strerror(errno)
    //         << "]";
    return false;
  }
  return true;
}

bool FileUtil::CreateDir(const string& dir) {
  int ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
  if (ret != 0) {
    // LOG_WARN << "failed to create dir. [dir: " << dir
    //         << "] [err: " << strerror(ret) << "]";
    return false;
  }
  return true;
}

bool FileUtil::get_file_content(const string& path, string* content) {
  if (content == NULL) {
    return false;
  }

  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    // LOG_WARN << "failed to open file: " << path;
    return false;
  }
  struct stat buf;
  if (::fstat(fd, &buf) != 0) {
    // LOG_WARN << "failed to lstat file: " << path;
    ::close(fd);
    return false;
  }

  size_t fsize = buf.st_size;
  content->resize(fsize);
  char* data = const_cast<char*>(content->data());
  int size = 0;
  size_t has_read = 0;
  do {
    size = ::read(fd, data + has_read, fsize - has_read);
    if (size < 0) {
      // LOG_WARN << "failed to read file: " << path;
      ::close(fd);
      return false;
    }
    has_read += size;
  } while (size > 0);

  ::close(fd);
  return true;
}

bool FileUtil::ReadLines(const string& path, vector<string>* lines) {
  std::ifstream fin(path);
  RETURN_VAL_IF2(!fin.good(),
                 false);  // LOG_ERROR << "Failed to open path: " << path;

  RETURN_VAL_IF2(lines == nullptr, false);

  string line;
  while (std::getline(fin, line)) {
    lines->push_back(line);
  }
  return true;
}

std::string FileUtil::RemoveFileSuffix(std::string filename) {
  int first_index = filename.find_last_of("/");
  size_t last_index = filename.find_last_of(".");
  if (last_index == std::string::npos) {
    last_index = filename.length();
  }
  std::string raw_name =
      filename.substr(first_index + 1, last_index - first_index - 1);
  return raw_name;
}

int FileUtil::get_file_list(vector<string>& files, const string path,
                            const string suffix) {
  return get_file_list(path, suffix, &files);
}

int FileUtil::get_file_list(const std::string& path,
                            std::vector<std::string>* files) {
  return get_file_list(path, "", files);
}

int FileUtil::get_file_list(const std::string& path, const std::string& suffix,
                            std::vector<std::string>* files) {
  if (!Exists(path)) {
    // LOG_INFO << path << " not exist.";
    return FAIL;
  }

  boost::filesystem::recursive_directory_iterator itr(path);
  while (itr != boost::filesystem::recursive_directory_iterator()) {
    try {
      if (suffix.empty() ||
          boost::algorithm::ends_with(itr->path().string(), suffix)) {
        files->push_back(itr->path().string());
      }
      ++itr;
    }
    catch (const std::exception& ex) {
      // LOG_WARN << "Caught execption: " << ex.what();
      continue;
    }
  }
  return SUCC;
}

string FileUtil::get_absolute_path(const string& prefix,
                                   const string& relative_path) {
  if (relative_path.empty()) {
    return prefix;
  }

  if (prefix.empty()) {
    return relative_path;
  }

  string result = prefix;

  if (relative_path[0] == '/') {
    return relative_path;
  }

  if (prefix[prefix.length() - 1] != '/') {
    result.append("/");
  }
  return result.append(relative_path);
}

void FileUtil::get_file_name(const string& file, string* name) {
  size_t pos_left = file.find_last_of('/');
  size_t pos_right = file.find_last_of('.');
  if (pos_right == string::npos) {
    *name = file.substr(pos_left + 1);
  } else {
    *name = file.substr(pos_left + 1, pos_right - pos_left - 1);
  }
}

bool FileUtil::CompareFileByDigital(const string& file_left,
                                    const string& file_right) {
  return CompareFile(file_left, file_right, FCT_DIGITAL);
}

bool FileUtil::CompareFileByLexicographical(const string& file_left,
                                            const string& file_right) {
  return CompareFile(file_left, file_right, FCT_LEXICOGRAPHICAL);
}

// private functions

bool FileUtil::CompareFile(const string& file_left, const string& file_right,
                           FileCompareType type) {
  string name_left;
  get_file_name(file_left, &name_left);
  string name_right;
  get_file_name(file_right, &name_right);

  switch (type) {
    case FCT_DIGITAL:
      return atoll(name_left.c_str()) < atoll(name_right.c_str());
    case FCT_LEXICOGRAPHICAL:
      return std::lexicographical_compare(name_left.begin(), name_left.end(),
                                          name_right.begin(), name_right.end());
    default:
      // LOG_ERROR << "Unknown compare type!";
      return false;
  }

  return true;
}

int FileUtil::NumLines(const std::string& filename) {
  std::ifstream ifs(filename.c_str());
  return ifs.good() ? count(istreambuf_iterator<char>(ifs),
                            istreambuf_iterator<char>(), '\n') +
                          1
                    : -1;
}

}  // namespace cybertron
