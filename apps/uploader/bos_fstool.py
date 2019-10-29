#!/usr/bin/env python

"""Tool for downloading/uploading files from/to BOS"""

import base64
import optparse

from Crypto.Cipher import DES

import fueling.common.s3_utils as s3_utils


def decrypt(ciber_text, passwd):
    """Decrypt a ciber text"""
    des = DES.new(passwd, DES.MODE_ECB)
    return des.decrypt(base64.b64decode(ciber_text))


def get_aws_keys():
    """Validate keys by decrypting from cibered keys"""
    passwd = ''
    aws_ak = decrypt('5eJsLHCWBfNANHv57FBa1ADoeD/34zoVBjzK2eIGUqc=', passwd)
    aws_sk = decrypt('qZU1bTVmWihKuYRfOoOm7yofoTzkW+t9TgvWoxr64zA=', passwd)
    return aws_ak, aws_sk


def main():
    """Main function"""
    parser = optparse.OptionParser()
    # Disable downloading from external for now, need to think about permissions
    # parser.add_option("-o", "--oper", help="specify the operation, must be get or put")
    parser.add_option("-s", "--src", help="specify the source")
    parser.add_option("-d", "--dst", help="specify the destination")
    (opts, _args) = parser.parse_args()
    if opts.src is None or opts.dst is None:
        parser.print_help()
        return

    bucket = 'apollo-platform'
    s3_utils.upload_file(bucket, opts.src, opts.dst, get_aws_keys())


if __name__ == '__main__':
    main()
