#!/usr/bin/env python3
# coding=utf-8

"""
flask_uuap_sso
==================

This module provides UUAP SSO authentication for Flask routes.

:copyright: (C) 2015-12 by dengzhaoqun created.
:copyright: (C) 2018-10 by wangpanjun update.
"""

import re
import requests
import functools
import flask
import hashlib
import time

import application


class SSO(object):
    def __init__(self, host=None, port=None, app_key=None):
        """
        :param host: CAS address, example: http://itebeta.baidu.com:8100
        :param port: port num
        :param app_key: appKey
        :return:
        """
        self.host = host
        self.port = port
        self.app_key = app_key
        self.validate_error_callback = None

    @property
    def uri(self):
        """
        host:port
        :return:
        """
        if self.port:
            return '%s:%s' % (self.host, self.port)
        return self.host

    def update_config(self, host=None, port=None, app_key=None):
        """
        Update configure.
        :param host: CAS address, example: http://itebeta.baidu.com:8100
        :param port: port num
        :param app_key: appKey
        :return:
        """
        self.host = host if host else self.host
        self.port = port if host else self.port
        self.app_key = app_key if host else self.app_key

    def error_handler(self, f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            res = f(*args, **kwargs)
            if type(res) == str:
                res = flask.make_response(res)
                res.status_code = 401
            return res

        self.validate_error_callback = decorated
        return decorated

    def _sso(self):
        """sso 登陆"""
        flask.flash('No ticket, redirect to UUAP ...')
        url = '%s/login?service=%s&appKey=%s&version=v2' % (
            self.host, flask.request.base_url, self.app_key)
        return flask.redirect(url)

    @classmethod
    def _get_sign(cls, data):
        """签名"""
        before_sha256 = functools.reduce(lambda x, y: '%s%s' % (x, y[-1]),
                                         sorted(data.items(), key=lambda x: x[0]), '')
        before_sha256 = '%s%s' % (before_sha256, application.app.config.get("UUAP_SECRET_KEY"))
        hash_256 = hashlib.sha256()
        hash_256.update(before_sha256.encode('utf-8'))
        return hash_256.hexdigest()

    def _get_s_token(self, ticket):
        """
        根据ticket 获取 token
        :param ticket:
        :return:
        """
        url = '%s/sTokenDecrypt' % self.uri
        timestamp = int(time.time())

        data = {
            'appKey': application.app.config.get("UUAP_APP_KEY"),
            'encryptedSToken': ticket,
            'timestamp': timestamp
        }

        data['sign'] = self._get_sign(data)
        response = requests.post(url, data=data)
        res = response.json()
        if res['code'] == 200 and res['msg'] == 'success':
            return res['result']

    def _validate_session(self, p_token, s_token):
        """
        验证token(p/s),获取用户信息
        :param p_token:
        :param s_token:
        :return:
        """
        url = '%s/session/validate' % self.uri
        data = dict(
            pToken=p_token,
            sToken=s_token,
            appKey=application.app.config.get("UUAP_APP_KEY"),
            timestamp=int(time.time())
        )
        sign = self._get_sign(data)
        data['sign'] = sign
        response = requests.post(url, data=data)
        res = response.json()
        if res['code'] == 200 and res['msg'] == 'success':
            return res['result']

    def login_required(self, f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            """
            1.检查session中username是否存在，如果存在表示登陆，否则进入【2】
            2.检查p_token和s_token是否同时存在，同时存在进入【3】，否则进入【4】
            3.验证p_token和s_token并获取用户信息，username写入session，返回response
            4.跳转uu_ap登陆，获取返回的ticket 进入步骤【5】
            5.通过ticket获取s_token 进入步骤【3】
            6.验证不通过跳转到验证失败页面
            :param args:
            :param kwargs:
            :return:
            """
            flag = False
            username = flask.session.get('username')
            if username:
                return f(*args, **kwargs)
            p_token = flask.request.cookies.get(application.app.config.get("UUAP_P_TOKEN"), '')
            s_token = flask.request.cookies.get(application.app.config.get("UUAP_S_TOKEN"), '')
            if not p_token or not s_token:
                url = flask.request.url
                if 'ticket=' not in url:
                    return self._sso()
                ticket = re.match(r".*\?ticket=(.*)$", url)
                s_token = self._get_s_token(ticket.group(1))
                if not s_token:
                    return self.validate_error_callback()
                flag = True
            user = self._validate_session(p_token, s_token)
            if not user:
                return self.validate_error_callback()
            flask.session['username'] = user['username']
            resp = f(*args, **kwargs)
            if flag:
                resp.set_cookie(application.app.config.get("UUAP_S_TOKEN"), s_token,
                                application.app.config.get("TOKEN_TIMEOUT"))
            return resp

        return decorated
