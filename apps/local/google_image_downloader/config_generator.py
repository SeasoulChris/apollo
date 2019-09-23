#!/usr/bin/env python

import json
import random
import sys


KEYWORDS = [
    'emergency vehicle police',
    'police vehicle',
    'police car',
    'emergency vehicle fire',
    'fire fighter vehicle',
    'fire vehicle',
    'fire engine',
    'emergency vehicle ambulance',
    'ambulance vehicle',
    'emergency vehicle',
]

# https://google-images-download.readthedocs.io/en/latest/arguments.html
# COLOR = [red, orange, yellow, green, teal, blue, purple, pink, white, gray, black, brown]

LOCATION_KEYWORDS = [
    '',
    'Bay Area',
    'California',
    'San Francisco',
    'San Jose',
    'Sunnyvale',
    'United States',
]

COLOR_TYPES = [
    'full-color',
    # black-and-white, transparent
]

SIZES = [
    # '>400*300',
    # '>640*480',
    '>800*600',
    '>1024*768',
]

ASPECT_RATIOS = [
    'tall',
    'square',
    'wide',
    # 'panoramic',
]

TYPES = [
    'photo',
    # face, clip-art, line-drawing, animated.
]

FORMATS = [
    'jpg',
    'png',
    # 'bmp', 'gif', 'svg', 'webp', 'ico', 'raw'
]


def main(argv):
    records = []
    for keyword in KEYWORDS:
        for location in LOCATION_KEYWORDS:
            if location:
                location = ' "%s"' % location
            for color_type in COLOR_TYPES:
                for size_opt in SIZES:
                    for aspect_ratio in ASPECT_RATIOS:
                        for type_opt in TYPES:
                            for format_opt in FORMATS:
                                records.append({
                                    'keywords': keyword + location,
                                    'color_type': color_type,
                                    'size': size_opt,
                                    'aspect_ratio': aspect_ratio,
                                    'type': type_opt,
                                    'format': format_opt,
                                })
    items = len(records)
    print('Generated {} config items which will download {} images.'.format(items, items * 100))
    random.shuffle(records)
    with open('google-image-downloader-conf.json', 'w') as fout:
        fout.write(json.dumps({'Records': records}, indent=2))


if __name__ == '__main__':
    main(sys.argv)
