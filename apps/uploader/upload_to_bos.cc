/******************************************************************************
 * Copyright 2019 The Apollo Authors. All Rights Reserved.
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

#include "cyber/common/file.h"
#include "gflags/gflags.h"
#include "modules/common/util/string_util.h"

DEFINE_string(from, "", "Local file path.");
DEFINE_string(to, "", "BOS file path.");

DEFINE_string(bos_bucket, "", "BOS bucket name.");
DEFINE_string(bos_access_key, "", "BOS access key.");
DEFINE_string(bos_secret_key, "", "BOS secret key.");
DEFINE_string(bos_mount_dir, "/mnt/bos_rw", "BOS mount directory.");

namespace apollo {
namespace data {
namespace fuel {

int UploadToBOS(const std::string& local_path, const std::string& bos_path) {
  CHECK(cyber::common::PathExists(local_path));

  using apollo::common::util::StrCat;
  // Step 1: Mount BOS.
  const std::string mount_bos_cmd = StrCat(
      StrCat("sudo mkdir -p \"", FLAGS_bos_mount_dir, "\""), " && "
      "sudo /usr/local/bin/bosfs ", FLAGS_bos_bucket, " ", FLAGS_bos_mount_dir,
      " -o allow_other,logfile=/tmp/bos.log,endpoint=http://bj.bcebos.com,"
      "ak=", FLAGS_bos_access_key, ",sk=", FLAGS_bos_secret_key);

  // Step 2: Copy local file to BOS.
  const std::string target_path = cyber::common::GetAbsolutePath(
      FLAGS_bos_mount_dir, FLAGS_to);
  const std::string copy_cmd = StrCat(
      "sudo mkdir -p \"$(dirname \"", target_path, "\")\" && "
      "sudo cp -n \"", FLAGS_from, "\" \"", target_path, "\"");

  // Step 3: Unmount BOS.
  const std::string umount_cmd = StrCat(
      "sudo fusermount -u \"", FLAGS_bos_mount_dir, "\"");

  // Action.
  const std::string command = StrCat(
      "bash -c '", mount_bos_cmd, " && ", copy_cmd, " && ", umount_cmd, "'");
  return system(command.c_str());
}

}  // namespace fuel
}  // namespace data
}  // namespace apollo

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  CHECK(!FLAGS_from.empty() && !FLAGS_to.empty()) <<
      "Please specify paths: --from=<path> --to=<path>";
  CHECK(!FLAGS_bos_bucket.empty() &&
        !FLAGS_bos_access_key.empty() && !FLAGS_bos_secret_key.empty()) <<
      "Please specify BOS credentials: --bos_bucket=<bucket> "
      "--bos_access_key=<access> --bos_secret_key=<secret>";
  return apollo::data::fuel::UploadToBOS(FLAGS_from, FLAGS_to);
}
