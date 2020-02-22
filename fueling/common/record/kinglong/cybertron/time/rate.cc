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
// #include "cybertron/common/logger.h"
#include "fueling/common/record/kinglong/cybertron/time/rate.h"

namespace cybertron {

Rate::Rate(double frequency)
    : start_(Time::Now()),
      expected_cycle_time_(1.0 / frequency),
      actual_cycle_time_(0.0) {}

Rate::Rate(uint64_t nanoseconds)
    : start_(Time::Now()),
      expected_cycle_time_(int32_t(nanoseconds/1000000000ull), 
            int32_t(nanoseconds%1000000000ull)),
      actual_cycle_time_(0.0) {}

Rate::Rate(const Duration& d)
    : start_(Time::Now()),
      expected_cycle_time_(d.sec, d.nsec),
      actual_cycle_time_(0.0) {}

bool Rate::sleep() {
  Time expected_end = start_ + expected_cycle_time_;

  Time actual_end = Time::Now();

  // detect backward jumps in time
  if (actual_end < start_) {
    expected_end = actual_end + expected_cycle_time_;
  }

  // calculate the time we'll sleep for
  Duration sleep_time = expected_end - actual_end;

  // set the actual amount of time the loop took in case the user wants to kNow
  actual_cycle_time_ = actual_end - start_;

  // make sure to reset our start time
  start_ = expected_end;

  // if we've taken too much time we won't sleep
  if (sleep_time < Duration(0.0)) {
    // if we've jumped forward in time, or the loop has taken more than a full
    // extra
    // cycle, reset our cycle
    if (actual_end > expected_end + expected_cycle_time_) {
      start_ = actual_end;
    }
    // return false to show that the desired rate was not met
    return false;
  }

  return Time::sleepUntil(expected_end);
}

void Rate::reset() { start_ = Time::Now(); }

Duration Rate::cycleTime() const { return actual_cycle_time_; }

WallRate::WallRate(double frequency)
    : start_(WallTime::Now()),
      expected_cycle_time_(1.0 / frequency),
      actual_cycle_time_(0.0) {}

WallRate::WallRate(const Duration& d)
    : start_(WallTime::Now()),
      expected_cycle_time_(d.sec, d.nsec),
      actual_cycle_time_(0.0) {}

bool WallRate::sleep() {
  WallTime expected_end = start_ + expected_cycle_time_;

  WallTime actual_end = WallTime::Now();

  // detect backward jumps in time
  if (actual_end < start_) {
    expected_end = actual_end + expected_cycle_time_;
  }

  // calculate the time we'll sleep for
  WallDuration sleep_time = expected_end - actual_end;

  // set the actual amount of time the loop took in case the user wants to kNow
  actual_cycle_time_ = actual_end - start_;

  // make sure to reset our start time
  start_ = expected_end;

  // if we've taken too much time we won't sleep
  if (sleep_time <= WallDuration(0.0)) {
    // if we've jumped forward in time, or the loop has taken more than a full
    // extra
    // cycle, reset our cycle
    if (actual_end > expected_end + expected_cycle_time_) {
      start_ = actual_end;
    }
    return true;
  }

  return sleep_time.sleep();
}

void WallRate::reset() { start_ = WallTime::Now(); }

WallDuration WallRate::cycleTime() const { return actual_cycle_time_; }
}
