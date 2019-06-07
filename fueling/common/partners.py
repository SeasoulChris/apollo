"""Partners."""
#!/usr/bin/env python

import collections


Partner = collections.namedtuple('Partner', ['name', 'email', 'phone'])

partners = {
    'apollo': Partner(name='Apollo', email='xiaoxiangquan@baidu.com', phone='+16502792727'),
    'apollo-evangelist': Partner(name='Evangelist', email=None, phone=None),
    'udelv2019': Partner(name='Udelv', email=None, phone=None),
}
