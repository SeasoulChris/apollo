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

#include <cmath>
#include <ctime>
#include <mutex>
#include <iomanip>
#include <stdexcept>
#include <limits>

#include <boost/io/ios_state.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

// #include "cybertron/common/logger.h"
// #include "cybertron/common/common.h"
#include "fueling/common/record/kinglong/cybertron/time/time.h"
#include "fueling/common/record/kinglong/cybertron/time/time_impl.h"

#define HAS_CLOCK_GETTIME (_POSIX_C_SOURCE >= 199309L)

namespace cybertron {

const Duration DURATION_MAX(std::numeric_limits<int32_t>::max(), 999999999);
const Duration DURATION_MIN(std::numeric_limits<int32_t>::min(), 0);

const Time TIME_MAX(std::numeric_limits<uint32_t>::max(), 999999999);
const Time TIME_MIN(0, 1);

static bool g_stopped(false);

static std::mutex g_sim_time_mutex;

static bool g_initialized(true);
static bool g_use_sim_time(false);
//const uint64_t SIM_BASE_TIME(1506823200000000000ull /* 2017-10-01 10:00:00 */);
const uint64_t SIM_BASE_TIME(946656000000000000ull /* 2000-01-01 00:00:00 */);
static Time g_sim_time(SIM_BASE_TIME);
//static Time g_sim_time(0, 0);

void cybertron_walltime(uint32_t& sec, uint32_t& nsec) {
#if HAS_CLOCK_GETTIME
  timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  sec = time.tv_sec;
  nsec = time.tv_nsec;
#else
  struct timeval timeofday;
  gettimeofday(&timeofday, NULL);
  sec = timeofday.tv_sec;
  nsec = timeofday.tv_usec * 1000;
#endif
}

void cybertron_monotime(uint32_t& sec, uint32_t& nsec) {
#if HAS_CLOCK_GETTIME
  timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  sec = time.tv_sec;
  nsec = time.tv_nsec;
#else
  cybertron_walltime(sec, nsec);
#endif
}

int cybertron_nanosleep(const uint32_t& sec, const uint32_t& nsec) {
  timespec req = {sec, nsec};
  return nanosleep(&req, NULL);
}

bool cybertron_wallsleep(uint32_t sec, uint32_t nsec) {
  timespec req = {sec, nsec};
  timespec rem = {0, 0};
  while (nanosleep(&req, &rem) && !g_stopped) {
    req = rem;
  }
  return !g_stopped;
}

bool Time::useSystemTime() { return !g_use_sim_time; }

bool Time::isSimTime() { return g_use_sim_time; }

bool Time::isSystemTime() { return !isSimTime(); }

Time Time::MonoTime() {
  if (!g_initialized) {
    throw TimeNotInitializedException();
  }

  if (g_use_sim_time) {
    std::lock_guard<std::mutex> lock(g_sim_time_mutex);
    Time t = g_sim_time;
    return t;
  }

  Time t;
  cybertron_monotime(t.sec, t.nsec);
  return t;
}

Time Time::Now() {
  if (!g_initialized) {
    throw TimeNotInitializedException();
  }

  if (g_use_sim_time) {
    std::lock_guard<std::mutex> lock(g_sim_time_mutex);
    Time t = g_sim_time;
    return t;
  }

  Time t;
  cybertron_walltime(t.sec, t.nsec);

  return t;
}

void Time::setNow(const Time& new_now) {
  std::lock_guard<std::mutex> lock(g_sim_time_mutex);

  g_sim_time = new_now;
  g_use_sim_time = true;
}

void Time::init(bool use_sim_time) {
  g_stopped = false;
  g_use_sim_time = use_sim_time;
  g_initialized = true;

  double base_time = 0.0;
  READ_CONF_WITH_DEFAULT("simulator", "base_time", base_time, 0.0);
  if (base_time > 0.0) {
    g_sim_time = Time(base_time);
    // LOG_INFO << "Set sim base time: " << g_sim_time.ToNanosecond() / 1000000
    //          << " ms.";
  }
}

void Time::shutdown() { g_stopped = true; }

bool Time::isValid() { return (!g_use_sim_time) || !g_sim_time.isZero(); }

bool Time::waitForValid() { return waitForValid(cybertron::WallDuration()); }

bool Time::waitForValid(const cybertron::WallDuration& timeout) {
  cybertron::WallTime start = cybertron::WallTime::Now();
  while (!isValid() && !g_stopped) {
    cybertron::WallDuration(0.01).sleep();

    if (timeout > cybertron::WallDuration(0, 0) &&
        (cybertron::WallTime::Now() - start > timeout)) {
      return false;
    }
  }

  if (g_stopped) {
    return false;
  }

  return true;
}

Time Time::fromBoost(const boost::posix_time::ptime& t) {
  boost::posix_time::time_duration diff = t - boost::posix_time::from_time_t(0);
  return Time::fromBoost(diff);
}

Time Time::fromBoost(const boost::posix_time::time_duration& d) {
  Time t;
  t.sec = d.total_seconds();
#if defined(BOOST_DATE_TIME_HAS_NANOSECONDS)
  t.nsec = d.fractional_seconds();
#else
  t.nsec = d.fractional_seconds() * 1000;
#endif
  return t;
}

std::ostream& operator<<(std::ostream& os, const Time& rhs) {
  boost::io::ios_all_saver s(os);
  os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Duration& rhs) {
  boost::io::ios_all_saver s(os);
  if (rhs.sec >= 0 || rhs.nsec == 0) {
    os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
  } else {
    os << (rhs.sec == -1 ? "-" : "") << (rhs.sec + 1) << "." << std::setw(9)
       << std::setfill('0') << (1000000000 - rhs.nsec);
  }
  return os;
}

bool Time::sleepUntil(const Time& end) {
  if (Time::useSystemTime()) {
    Duration d(end - Time::Now());
    if (d > Duration(0)) {
      return d.sleep();
    }

    return true;
  } else {
    Time start = Time::Now();
    while (!g_stopped && (Time::Now() < end)) {
      cybertron_nanosleep(0, 1000000);
      if (Time::Now() < start) {
        return false;
      }
    }

    return true;
  }
}

bool WallTime::sleepUntil(const WallTime& end) {
  WallDuration d(end - WallTime::Now());
  if (d > WallDuration(0)) {
    return d.sleep();
  }

  return true;
}

bool Duration::sleep() const {
  if (Time::useSystemTime()) {
    return cybertron_wallsleep(sec, nsec);
  } else {
    Time start = Time::Now();
    Time end = start + *this;
    if (start.isZero()) {
      end = TIME_MAX;
    }

    while (!g_stopped && (Time::Now() < end)) {
      cybertron_wallsleep(0, 1000000);

      if (start.isZero()) {
        start = Time::Now();
        end = start + *this;
      }

      if (Time::Now() < start) {
        return false;
      }
    }

    return true;
  }
}

std::ostream& operator<<(std::ostream& os, const WallTime& rhs) {
  boost::io::ios_all_saver s(os);
  os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
  return os;
}

WallTime WallTime::Now() {
  WallTime t;
  cybertron_walltime(t.sec, t.nsec);

  return t;
}

std::ostream& operator<<(std::ostream& os, const WallDuration& rhs) {
  boost::io::ios_all_saver s(os);
  if (rhs.sec >= 0 || rhs.nsec == 0) {
    os << rhs.sec << "." << std::setw(9) << std::setfill('0') << rhs.nsec;
  } else {
    os << (rhs.sec == -1 ? "-" : "") << (rhs.sec + 1) << "." << std::setw(9)
       << std::setfill('0') << (1000000000 - rhs.nsec);
  }
  return os;
}

bool WallDuration::sleep() const { return cybertron_wallsleep(sec, nsec); }

void normalizeSecNSec(uint64_t& sec, uint64_t& nsec) {
  uint64_t nsec_part = nsec % 1000000000UL;
  uint64_t sec_part = nsec / 1000000000UL;

  if (sec + sec_part > UINT_MAX)
    throw std::runtime_error("Time is out of dual 32-bit range");

  sec += sec_part;
  nsec = nsec_part;
}

void normalizeSecNSec(uint32_t& sec, uint32_t& nsec) {
  uint64_t sec64 = sec;
  uint64_t nsec64 = nsec;

  normalizeSecNSec(sec64, nsec64);

  sec = (uint32_t)sec64;
  nsec = (uint32_t)nsec64;
}

void normalizeSecNSecUnsigned(int64_t& sec, int64_t& nsec) {
  int64_t nsec_part = nsec % 1000000000L;
  int64_t sec_part = sec + nsec / 1000000000L;
  if (nsec_part < 0) {
    nsec_part += 1000000000L;
    --sec_part;
  }

  if (sec_part < 0 || sec_part > UINT_MAX)
    throw std::runtime_error("Time is out of dual 32-bit range");

  sec = sec_part;
  nsec = nsec_part;
}

template class TimeBase<Time, Duration>;
template class TimeBase<WallTime, WallDuration>;
}
