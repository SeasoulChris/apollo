#!/usr/bin/env python3
"""
uuapsso module global variable
"""

import application
from uuap_sso import uuapsso


auth = uuapsso.SSO(host=application.app.config.get("UUAP_HOST"),
                   port=application.app.config.get("UUAP_POST"),
                   app_key=application.app.config.get("UUAP_APP_KEY"))