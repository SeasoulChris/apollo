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

#include "fueling/common/record/kinglong/cybertron/recorder/compress.h"

using fueling::common::record::kinglong::proto::cybertron::CompressType;

namespace cybertron {

CompressBase::CompressBase() {}

CompressBase::~CompressBase() {}

// BZ2Compress::BZ2Compress()
//     : _verbosity(0), _block_size_100k(9), _work_factor(30) {}

// BZ2Compress::BZ2Compress(int verbosity, int block_size_100k, int work_factor)
//     : _verbosity(verbosity),
//       _block_size_100k(block_size_100k),
//       _work_factor(work_factor) {}

// int BZ2Compress::compress(const std::string& raw, std::string& compressed) {
//   if (raw.empty()) {
//     LOG_INFO << "raw data is empty.";
//     return FAIL;
//   }

//   // init buffer for output data
//   unsigned int buffer_len = raw.length() * 1.01 + 600;
//   std::vector<char> buffer(buffer_len);

//   int result = BZ2_bzBuffToBuffCompress(
//       buffer.data(), &buffer_len, const_cast<char*>(raw.data()), raw.length(),
//       _block_size_100k, _verbosity, _work_factor);
//   ERROR_AND_RETURN_VAL_IF(result == BZ_OUTBUFF_FULL, FAIL, CYBERTRON_ERROR, BZ2_COMPRESS_CAPACITY_ERROR);
//   ERROR_AND_RETURN_VAL_IF(result != BZ_OK, FAIL, CYBERTRON_ERROR, BZ2_COMPRESS_ERROR);
//   compressed.assign(buffer.data(), buffer_len);
//   return SUCC;
// }

// int BZ2Compress::Decompress(const std::string& compressed, std::string& raw) {
//   if (compressed.empty()) {
//     LOG_INFO << "compressed data is empty.";
//     return FAIL;
//   }

//   if (raw.capacity() <= 0) {
//     LOG_ERROR << CYBERTRON_ERROR << BZ2_DECOMPRESS_CAPACITY_ERROR << " raw string out buffer is empty. please reserve first.";
//     return FAIL;
//   }

//   // init buffer for output data
//   unsigned int buffer_len = raw.capacity();
//   std::vector<char> buffer(buffer_len);
//   int result = BZ2_bzBuffToBuffDecompress(buffer.data(), &buffer_len,
//                                           const_cast<char*>(compressed.data()),
//                                           compressed.length(), 0, _verbosity);
//   ERROR_AND_RETURN_VAL_IF(result == BZ_OUTBUFF_FULL, FAIL, CYBERTRON_ERROR, BZ2_DECOMPRESS_CAPACITY_ERROR);
//   ERROR_AND_RETURN_VAL_IF(result == BZ_DATA_ERROR, FAIL, CYBERTRON_ERROR, BZ2_DECOMPRESS_RAWDATA_ERROR);
//   ERROR_AND_RETURN_VAL_IF(result != BZ_OK, FAIL, CYBERTRON_ERROR, BZ2_DECOMPRESS_ERROR);
//   raw.assign(buffer.data(), buffer_len);
//   return SUCC;
// }

// LZ4Compress::LZ4Compress() {}

// int LZ4Compress::compress(const std::string& raw, std::string& compressed) {
//   if (raw.empty()) {
//     LOG_INFO << "raw data is empty.";
//     return FAIL;
//   }

//   // init buffer for output data
//   unsigned int buffer_len = raw.length();
//   std::vector<char> buffer(buffer_len);

//   int result =
//       LZ4_compress(const_cast<char*>(raw.data()), buffer.data(), raw.length());
//   ERROR_AND_RETURN_VAL_IF(result <= 0, FAIL, CYBERTRON_ERROR, LZ4_COMPRESS_ERROR);
//   compressed.resize(result);
//   compressed.assign(buffer.data(), result);
//   return SUCC;
// }

// int LZ4Compress::Decompress(const std::string& compressed, std::string& raw) {
//   if (compressed.empty()) {
//     LOG_INFO << "compressed data is empty.";
//     return FAIL;
//   }

//   if (raw.capacity() <= 0) {
//     LOG_ERROR << CYBERTRON_ERROR << LZ4_DECOMPRESS_CAPACITY_ERROR << " raw string out buffer is empty. please reserve first.";
//     return FAIL;
//   }

//   // init buffer for output data
//   unsigned int buffer_len = raw.capacity();
//   char buffer[buffer_len];
//   // std::vector<char> buffer(buffer_len);
//   int result = LZ4_decompress_fast(const_cast<char*>(compressed.data()),
//                                    (char*)buffer, buffer_len);
//   ERROR_AND_RETURN_VAL_IF(result <= 0, FAIL, CYBERTRON_ERROR, LZ4_DECOMPRESS_ERROR);
//   raw.assign(buffer, buffer_len);
//   return SUCC;
// }

CompressBase::SharedPtr CompressFactory::Create(
    const CompressType& compress) {
  switch (compress) {
    case fueling::common::record::kinglong::proto::cybertron::COMPRESS_NONE:
      return nullptr;
    // case cybertron::proto::COMPRESS_BZ2:
    //   return BZ2Compress::make_shared();
    // case cybertron::proto::COMPRESS_LZ4:
    //   return LZ4Compress::make_shared();
    default:
      return nullptr;
  }
  return nullptr;
}

}  // namespace cybertron
