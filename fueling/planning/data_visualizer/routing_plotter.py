#!/usr/bin/env python3
from shapely.geometry import LineString
from shapely.geometry import Point

import fueling.common.logging as logging
from fueling.planning.map_reader.map_reader import MapReader


def plot(routing_response, ax):
    map_reader = MapReader()
    # colors = ["b", "r", "g", "k"]
    colors = ["k"]
    cnt = 1
    for road in routing_response.road:
        ind = cnt % len(colors)
        cnt += 1
        c = colors[ind]
        passage_cnt = 1
        # passage_pt = [".", "o", "v"]
        for passage in road.passage:
            # pind = passage_cnt % len(passage_pt)
            # pt = passage_pt[pind]
            passage_cnt += 1
            for segment in passage.segment:
                # logging.info(segment.id)
                coords = map_reader.lane_id_to_coords(segment.id)
                if coords is None:
                    logging.info("didn't found lane id: " + segment.id)
                else:
                    x = []
                    y = []
                    for coord in coords:
                        x.append(coord[0])
                        y.append(coord[1])
                    # ax.plot(x, y, lw=0.5, c=c, marker=pt)
                    ax.plot(x, y, lw=8, c=c, alpha=0.1)


def plot_with_loc(routing_response, ax, fist_loc, last_loc):
    started = False
    ended = False

    x = fist_loc.pose.position.x
    y = fist_loc.pose.position.y
    p_start = Point(x, y)
    ax.plot([x], [y], 'bo')
    logging.info("start x =" + str(x))
    logging.info("start y =" + str(y))

    x = last_loc.pose.position.x
    y = last_loc.pose.position.y
    p_end = Point(x, y)
    ax.plot([x], [y], 'bo')
    logging.info("end x =" + str(x))
    logging.info("end y =" + str(y))

    map_reader = MapReader()
    # colors = ["b", "r", "g", "k"]
    # colors = ["k"]
    cnt = 1
    for road in routing_response.road:
        # ind = cnt % len(colors)
        cnt += 1
        # c = colors[ind]
        passage_cnt = 1
        # passage_pt = [".", "o", "v"]
        for passage in road.passage:
            # pind = passage_cnt % len(passage_pt)
            # pt = passage_pt[pind]
            passage_cnt += 1
            passage_string = []
            for segment in passage.segment:
                # logging.info(segment.id)
                coords = map_reader.lane_id_to_coords(segment.id)
                if coords is None:
                    logging.info("didn't found lane id: " + segment.id)
                else:
                    for coord in coords:
                        passage_string.append((coord[0], coord[1]))

            string = LineString(passage_string)
            if not started:
                if p_start.distance(string) < 25:
                    plot_passage(ax, passage_string)
                    started = True

            if started and not ended:
                if p_end.distance(string) < 10:
                    ended = True
                plot_passage(ax, passage_string)

            if started and ended:
                break


def plot_passage(ax, passage):
    x = []
    y = []
    for p in passage:
        x.append(p[0])
        y.append(p[1])
    ax.plot(x, y, lw=8, c="k", alpha=0.1)
