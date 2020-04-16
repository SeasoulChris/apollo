#!/usr/bin/env python3

import fueling.common.logging as logging
from fueling.planning.map_reader.map_reader import MapReader


def plot(routing_response, ax):
    map_reader = MapReader()
    colors = ["b", "r", "g", "k"]
    cnt = 1
    for road in routing_response.road:
        ind = cnt % len(colors)
        cnt += 1
        c = colors[ind]
        passage_cnt = 1
        passage_pt = [".", "o", "v"]
        for passage in road.passage:
            pind = passage_cnt % len(passage_pt)
            pt = passage_pt[pind]
            passage_cnt += 1
            for segment in passage.segment:
                logging.info(segment.id)
                coords = map_reader.lane_id_to_coords(segment.id)
                if coords is None:
                    logging.info("didn't found lane id: " + segment.id)
                else:
                    x = []
                    y = []
                    for coord in coords:
                        x.append(coord[0])
                        y.append(coord[1])
                    ax.plot(x, y, lw=0.5, c=c, marker=pt)
