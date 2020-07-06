#!/usr/bin/env python3
"""
The entry module of the program
"""

import application
from common import filter
from views import index
from views import job
from views import statistics


# Register blueprint
application.app.register_blueprint(index.blue_index)
application.app.register_blueprint(job.blue_job)
application.app.register_blueprint(statistics.blue_statistics)


# Register filter
application.app.add_template_filter(filter.get_show_job_type, "show_type")
application.app.add_template_filter(filter.get_action, "show_action")
application.app.add_template_filter(filter.get_duration, "show_duration")
application.app.add_template_filter(filter.get_cn_action, "show_cn_action")
application.app.add_template_filter(filter.get_failure_cause, "show_failure_cause")
application.app.add_template_filter(filter.truncation_job_id, "show_id")

if __name__ == "__main__":
    application.app.run(host='0.0.0.0', port=8000, debug=True)
