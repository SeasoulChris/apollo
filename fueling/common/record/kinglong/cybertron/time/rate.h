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

#ifndef INCLUDE_CYBERTRON_TIME_RATE_H_
#define INCLUDE_CYBERTRON_TIME_RATE_H_

#include "fueling/common/record/kinglong/cybertron/time/time.h"
#include "fueling/common/record/kinglong/cybertron/time/cybertrontime_decl.h"

namespace cybertron {
class Duration;

class CYBERTIME_DECL Rate {
 public:
  Rate(double frequency);
  Rate(uint64_t nanoseconds);
  explicit Rate(const Duration&);
  bool sleep();
  void reset();
  Duration cycleTime() const;
  Duration expectedCycleTime() const { return expected_cycle_time_; }

 private:
  Time start_;
  Duration expected_cycle_time_, actual_cycle_time_;
};

class CYBERTIME_DECL WallRate {
 public:
  WallRate(double frequency);
  explicit WallRate(const Duration&);

  bool sleep();
  void reset();
  WallDuration cycleTime() const;

  WallDuration expectedCycleTime() const { return expected_cycle_time_; }

 private:
  WallTime start_;
  WallDuration expected_cycle_time_, actual_cycle_time_;
};
}

#endif
