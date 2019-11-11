import json
import sys

from fueling.planning.stability.grading.planning_stability_grader import PlanningStabilityGrader

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python dreamland.py record_path_file")
        sys.exit(1)
    fn = sys.argv[1]
    grader = PlanningStabilityGrader()
    grader.grade_record_file(fn)
    output = dict()
    output["lat_jerk_av"] = grader.lat_jerk_av_table
    output["lon_jerk_av"] = grader.lon_jerk_av_table
    print(json.dumps(output))
