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

// #include <bzlib.h>
// #include <lz4.h>

// #include "cybertron/common/common.h"
#include "fueling/common/record/kinglong/cybertron/common/macros.h"
#include "fueling/common/record/kinglong/proto/cybertron/record.pb.h"

namespace cybertron {

class CompressBase {
 public:
  SMART_PTR_DEFINITIONS_NOT_COPYABLE(CompressBase)
  CompressBase();
  virtual ~CompressBase();

  /**
   * \biref: compress the buffer string.
   * \param raw: the raw data chunk
   * \param comp: the compressed data of chunk
   * \returns : if success to compress, return 0, else lower than 0.
   */
  virtual int compress(const std::string& raw, std::string& compressed) = 0;

  /**
   * \biref: decompress the compressed data of string.
   * \param comp: the compressed data of chunk
   * \param raw: the raw data buffer, should reserve enough capacity bigger than
   * raw data length.
   * \returns : if success to decompress, return 0, else lower than 0.
   */
  virtual int Decompress(const std::string& compressed, std::string& raw) = 0;
};

// class BZ2Compress : public CompressBase {
//  public:
//   SMART_PTR_DEFINITIONS(BZ2Compress)
//   BZ2Compress();
//   virtual ~BZ2Compress() {}

//   BZ2Compress(int verbosity, int block_size_100k, int work_factor);

//   /**
//    * \returns : if success to decompress, return 0, else lower than 0.
//    *            error code defined in bzlib.h.
//    */
//   int compress(const std::string& raw, std::string& compressed) override;

//   /**
//    * \param raw: the raw data buffer, should reserve enough capacity bigger than
//    * raw data length.
//    * \returns : if success to decompress, return 0, else lower than 0.
//    *            error code defined in bzlib.h.
//    */
//   int Decompress(const std::string& compressed, std::string& raw) override;

//  private:
//   // level of debugging output (0-4; 0 default). 0 is silent, 4 is max verbose
//   // debugging output
//   int _verbosity;
//   // compression block size (1-9; 9 default). 9 is best compression, most memory
//   int _block_size_100k;
//   // compression behavior for worst case, highly repetitive data (0-250; 30
//   // default)
//   int _work_factor;
// };

// class LZ4Compress : public CompressBase {
//  public:
//   SMART_PTR_DEFINITIONS(LZ4Compress)
//   LZ4Compress();
//   virtual ~LZ4Compress() {}
//   LZ4Compress(int verbosity, int block_size_100k, int work_factor);
//   int compress(const std::string& raw, std::string& compressed) override;
//   int Decompress(const std::string& compressed, std::string& raw) override;

//  private:
//   int _verbosity;
//   int _block_size_100k;
//   int _work_factor;
// };

class CompressFactory {
 public:
  static CompressBase::SharedPtr Create(
      const fueling::common::record::kinglong::proto::cybertron::CompressType& compress);
};


}  // namespace cybertron
