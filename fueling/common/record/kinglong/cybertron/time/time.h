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

#ifndef INCLUDE_CYBERTRON_TIME_TIME_H_
#define INCLUDE_CYBERTRON_TIME_TIME_H_

#ifdef _MSC_VER

#pragma warning(disable : 4244)
#pragma warning(disable : 4661)
#endif

// #include <boost/math/special_functions/round.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "fueling/common/record/kinglong/cybertron/time/cybertrontime_decl.h"
#include "fueling/common/record/kinglong/cybertron/time/duration.h"
#include "fueling/common/record/kinglong/cybertron/time/rate.h"

#include <sys/time.h>

namespace boost {
namespace posix_time {
class ptime;
class time_duration;
}  // namespace posix_time
}  // namespace boost

namespace cybertron {

class Exception : public std::runtime_error {
 public:
  Exception(const std::string& what) : std::runtime_error(what) {}
};

class CYBERTIME_DECL TimeNotInitializedException : public Exception {
 public:
  TimeNotInitializedException()
      : Exception(
            "Cannot use cybertron::Time::Now() before the first NodeHandle has "
            "been created or cybertron::start() has been called.  "
            "If this is a standalone app or test that just uses "
            "cybertron::Time and does not communicate over ROS, you may also "
            "call cybertron::Time::init()") {}
};

class CYBERTIME_DECL NoHighPerformanceTimersException : public Exception {
 public:
  NoHighPerformanceTimersException()
      : Exception(
            "This windows platform does not "
            "support the high-performance timing api.") {}
};

CYBERTIME_DECL void normalizeSecNSec(uint64_t& sec, uint64_t& nsec);
CYBERTIME_DECL void normalizeSecNSec(uint32_t& sec, uint32_t& nsec);
CYBERTIME_DECL void normalizeSecNSecUnsigned(int64_t& sec, int64_t& nsec);

template <class T, class D>
class TimeBase {
 public:
  uint32_t sec, nsec;

  TimeBase() : sec(0), nsec(0) {}
  TimeBase(uint32_t _sec, uint32_t _nsec) : sec(_sec), nsec(_nsec) {
    normalizeSecNSec(sec, nsec);
  }
  explicit TimeBase(double t) { fromSec(t); }
  ~TimeBase() {}
  D operator-(const T& rhs) const;
  T operator+(const D& rhs) const;
  T operator-(const D& rhs) const;
  T& operator+=(const D& rhs);
  T& operator-=(const D& rhs);
  bool operator==(const T& rhs) const;
  inline bool operator!=(const T& rhs) const {
    return !(*static_cast<const T*>(this) == rhs);
  }
  bool operator>(const T& rhs) const;
  bool operator<(const T& rhs) const;
  bool operator>=(const T& rhs) const;
  bool operator<=(const T& rhs) const;

  double ToSecond() const { return (double)sec + 1e-9 * (double)nsec; };
  T& fromSec(double t) {
    sec = (uint32_t)floor(t);
    nsec = (uint32_t)std::round((t - sec) * 1e9);
    return *static_cast<T*>(this);
  }

  uint64_t ToNanosecond() const {
    return (uint64_t)sec * 1000000000ull + (uint64_t)nsec;
  }
  std::string ToString() const {
    time_t time_seconds = sec;
    struct tm* now_time = localtime(&time_seconds);
    std::ostringstream ostr;
    ostr << now_time->tm_year + 1900 << "-" << now_time->tm_mon + 1 << "-"
         << now_time->tm_mday << " " << now_time->tm_hour << ":"
         << now_time->tm_min << ":" << now_time->tm_sec << "." << nsec;
    return ostr.str();
  }
  T& fromNSec(uint64_t t);

  inline bool isZero() const { return sec == 0 && nsec == 0; }
  inline bool is_zero() const { return isZero(); }
  // boost::posix_time::ptime toBoost() const;
};

class CYBERTIME_DECL Time : public TimeBase<Time, Duration> {
 public:
  Time() : TimeBase<Time, Duration>() {}

  Time(uint32_t _sec, uint32_t _nsec) : TimeBase<Time, Duration>(_sec, _nsec) {}

  explicit Time(double t) { fromSec(t); }
  Time(uint64_t t) { fromNSec(t); }
  static Time Now();
  static Time MonoTime();
  static bool sleepUntil(const Time& end);

  static void init(bool use_sim_time = false);
  static void shutdown();
  static void setNow(const Time& new_now);
  static bool useSystemTime();
  static bool isSimTime();
  static bool isSystemTime();

  static bool isValid();
  static bool waitForValid();
  static bool waitForValid(const cybertron::WallDuration& timeout);

  // static Time fromBoost(const boost::posix_time::ptime& t);
  // static Time fromBoost(const boost::posix_time::time_duration& d);
};

extern CYBERTIME_DECL const Time TIME_MAX;
extern CYBERTIME_DECL const Time TIME_MIN;

class CYBERTIME_DECL WallTime : public TimeBase<WallTime, WallDuration> {
 public:
  WallTime() : TimeBase<WallTime, WallDuration>() {}

  WallTime(uint32_t _sec, uint32_t _nsec)
      : TimeBase<WallTime, WallDuration>(_sec, _nsec) {}

  explicit WallTime(double t) { fromSec(t); }

  static WallTime Now();

  static bool sleepUntil(const WallTime& end);

  static bool isSystemTime() { return true; }
};

class TimeoutChecker {
 public:
  TimeoutChecker() { _prev_time = Time::MonoTime().ToNanosecond(); }
  bool Check(uint64_t time_out, uint64_t* duration) {
    uint64_t curr_time = Time::MonoTime().ToNanosecond();
    *duration = curr_time - _prev_time;
    _prev_time = curr_time;
    return *duration < time_out;
  }

 private:
  uint64_t _prev_time = 0;
};

CYBERTIME_DECL std::ostream& operator<<(std::ostream& os, const Time& rhs);
CYBERTIME_DECL std::ostream& operator<<(std::ostream& os, const WallTime& rhs);
}  // namespace cybertron

#endif  // INCLUDE_CYBERTRON_TIME_TIME_H_
