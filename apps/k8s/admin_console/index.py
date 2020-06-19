#!/usr/bin/env python3
"""
The entry module of the program
"""

import application
from views import index
from views import job


# Register blueprint
application.app.register_blueprint(index.blue_index)
application.app.register_blueprint(job.blue_job)


if __name__ == "__main__":
    application.app.run()
