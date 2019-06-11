"""Partners."""
#!/usr/bin/env python

import collections


Partner = collections.namedtuple('Partner', ['name', 'email'])

partners = {
    'apollo': Partner(name='Apollo', email='xiaoxiangquan@baidu.com'),
    'apollo-evangelist': Partner(name='Evangelist', email='machao20@baidu.com'),
    'udelv2019': Partner(name='Udelv', email=None),
}
