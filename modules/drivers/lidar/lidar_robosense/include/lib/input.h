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

#ifndef CYBERTRON_ONBOARD_DRIVERS_SUTENG_INCLUDE_SUTENG_LIB_INPUT_H
#define CYBERTRON_ONBOARD_DRIVERS_SUTENG_INCLUDE_SUTENG_LIB_INPUT_H

#include <pcap.h>
#include <stdio.h>
#include <unistd.h>

#include "cyber/cyber.h"
#include "modules/drivers/lidar/lidar_robosense/include/lib/data_type.h"
#include "modules/drivers/lidar/lidar_robosense/proto/sensor_suteng.pb.h"

namespace autobot {
namespace drivers {
namespace robosense {

static const size_t FIRING_DATA_PACKET_SIZE = 1248;
static const size_t POSITIONING_DATA_PACKET_SIZE =
    1248;  // beike-256  velodyne-512;
static const size_t ETHERNET_HEADER_SIZE = 42;
static const int PCAP_FILE_END = -1;
static const int SOCKET_TIMEOUT = -2;
static const int RECIEVE_FAIL = -3;

/** @brief Pure virtual suteng input base class */
class Input {
 public:
  Input() {}
  virtual ~Input()  {}

  /** @brief Read one suteng packet.
   *
   * @param pkt points to SutengPacket message
   *
   * @returns 0 if successful,
   *          -1 if end of file
   *          > 0 if incomplete packet (is this possible?)
   */
  virtual int get_firing_data_packet(
      adu::common::sensor::suteng::SutengPacket* pkt, int time_zone,
      uint64_t _start_time) = 0;
  virtual int get_positioning_data_packtet(const NMEATimePtr& nmea_time) = 0;
  virtual void init()  {}
  virtual void init(uint32_t port) { (void)port; }

  bool flags = false;

 protected:
  // static uint64_t last_pkt_stamp;
  bool exract_nmea_time_from_packet(const NMEATimePtr& nmea_time,
                                    const uint8_t* bytes);
};
}  // namespace robosense
}  // namespace drivers
}  // namespace autobot

#endif  // CYBERTRON_ONBOARD_DRIVERS_SUTENG_INCLUDE_SUTENG_LIB_INPUT_H
