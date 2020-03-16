#!/usr/bin/env python
"""Localization Analyzer."""


class RoutingUpdateAnalyzer:
    def __init__(self):
        self.msgs_set = []
        self.msgs_set_has_routing = []
        self.debug = True

    def process(self, msgs):
        routing_available = False
        temp_msgs = []
        perception_msg_cnt = 0
        last_hmi = None

        for msg in msgs:
            if msg.topic == '/apollo/perception/obstacles':
                perception_msg_cnt += 1

            if msg.topic == "/apollo/routing_response_history":
                routing_available = True

            if msg.topic == "/apollo/hmi/status":
                last_hmi = msg

            if msg.topic == "/apollo/routing_response":
                if perception_msg_cnt > 80:
                    self.msgs_set.append(temp_msgs)
                    self.msgs_set_has_routing.append(routing_available)

                temp_msgs = []
                perception_msg_cnt = 0
                routing_available = True
                if last_hmi is not None:
                    temp_msgs.append(last_hmi)
            temp_msgs.append(msg)

        if perception_msg_cnt > 80:
            self.msgs_set.append(temp_msgs)
            self.msgs_set_has_routing.append(routing_available)

        if self.debug:
            for i in range(len(self.msgs_set_has_routing)):
                print(len(self.msgs_set[i]), self.msgs_set_has_routing[i])
        return self.msgs_set, self.msgs_set_has_routing


if __name__ == "__main__":
    import sys
    from cyber_py3.record import RecordReader

    fn = sys.argv[1]
    reader = RecordReader(fn)
    msgs = [msg for msg in reader.read_messages()]
    analyzer = RoutingAnalyzer()
    analyzer.analyze(msgs)
