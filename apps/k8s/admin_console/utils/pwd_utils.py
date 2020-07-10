"""
Some tool functions about password
"""

from hashlib import md5


def pwd_md5(params):
    """
    password MD5 encrypt
    """
    m = md5(b'dashqnnc248bda')
    m.update(params.encode("utf-8"))
    return m.hexdigest()
