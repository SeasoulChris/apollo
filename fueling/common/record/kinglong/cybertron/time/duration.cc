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

#include "fueling/common/record/kinglong/cybertron/time/duration.h"

namespace cybertron {

Duration::Duration(const Rate& rate)
    : DurationBase<Duration>(rate.expectedCycleTime().sec,
                             rate.expectedCycleTime().nsec) {}

WallDuration::WallDuration(const Rate& rate)
    : DurationBase<WallDuration>(rate.expectedCycleTime().sec,
                                 rate.expectedCycleTime().nsec) {}

void normalizeSecNSecSigned(int64_t& sec, int64_t& nsec) {
  int64_t nsec_part = nsec % 1000000000L;
  int64_t sec_part = sec + nsec / 1000000000L;
  if (nsec_part < 0) {
    nsec_part += 1000000000L;
    --sec_part;
  }

  if (sec_part < INT_MIN || sec_part > INT_MAX)
    throw std::runtime_error("Duration is out of dual 32-bit range");

  sec = sec_part;
  nsec = nsec_part;
}

void normalizeSecNSecSigned(int32_t& sec, int32_t& nsec) {
  int64_t sec64 = sec;
  int64_t nsec64 = nsec;

  normalizeSecNSecSigned(sec64, nsec64);

  sec = (int32_t)sec64;
  nsec = (int32_t)nsec64;
}

template class DurationBase<Duration>;
template class DurationBase<WallDuration>;
}
