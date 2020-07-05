#!/usr/bin/env python3
"""
The entry module of the program
"""

import application
from views import index
from views import job
from views import statistics


# Register blueprint
application.app.register_blueprint(index.blue_index)
application.app.register_blueprint(job.blue_job)
application.app.register_blueprint(statistics.blue_statistics)


if __name__ == "__main__":
    application.app.run(host='0.0.0.0', port=8000, debug=True)
