#!/usr/bin/env python3
"""
The entry module of the program
"""

import application
from common import filter
from middleware import before_request
from services import account_services
from services import job_services
from views import account
from views import admin
from views import index
from views import job
from views import statistics


application.api.add_resource(account_services.AccountService, "/account")
application.api.add_resource(job_services.JobService,
                             "/vehicle/<vehicle_sn>",
                             "/vehicle/<vehicle_sn>/jobs",
                             endpoint="vehicle")
application.app.before_request(before_request.process_request)

# Register blueprint
application.app.register_blueprint(admin.blue_admin)
application.app.register_blueprint(index.blue_index)
application.app.register_blueprint(job.blue_job)
application.app.register_blueprint(statistics.blue_statistics)
application.app.register_blueprint(account.blue_account)


# Register filter
application.app.add_template_filter(filter.get_show_job_type, "show_type")
application.app.add_template_filter(filter.get_action, "show_action")
application.app.add_template_filter(filter.get_duration, "show_duration")
application.app.add_template_filter(filter.get_cn_action, "show_cn_action")
application.app.add_template_filter(filter.get_failure_cause, "show_failure_cause")
application.app.add_template_filter(filter.truncation_job_id, "show_id")
application.app.add_template_filter(filter.get_en_action, "show_en_action")
application.app.add_template_filter(filter.get_account_show_action, "show_account_action")
application.app.add_template_filter(filter.get_account_show_region, "show_account_region")
application.app.add_template_filter(filter.get_account_show_status, "show_account_status")


if __name__ == "__main__":
    application.app.run(host='0.0.0.0', port=8000, debug=True)
