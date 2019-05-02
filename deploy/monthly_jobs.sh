#!/usr/bin/env bash

# Jobs that run once every month, starting at 5:00 of day 5.
# Crontab example: 0 5 5 * * /this/script.sh

set -e

# Preapre: Goto fuel root.
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Add your job here.
