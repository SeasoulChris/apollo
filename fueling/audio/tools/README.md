* Initial version of audio labeler, still may have dup code and need to clean & refactor
* No args, just modify the global variables in the file before run

## Arguments

* DIRECTION_FROM_RAW_DATA = False
  * To extract direction detection data directly from the audio channel, set this to False
  * Otherwise, will collection raw data from the microphone channel and estimate direction from python code -- ./direction_detection.py
* TARGET_OBSTACLE_ID = [1165, 1174]
  * All obstacle IDs of EV
  * Currently, obstacle id might change when the perception is not stable
* RECORD_PREFIX = 'public-test/2020/2020-08-24/2020-08-24-14-58-13'
* OUTPUT_DIR = 'modules/audio/metrics'
* EXTRINSIC_FILE_PATH = 'modules/audio/conf/respeaker_extrinsics.yaml'
  * Usually no need to change except that car is changed

## To do

* Still need testing on the cloud side for the audio mode (DIRECTION_FROM_RAW_DATA = False)
* Clean code and remove dup code
* Change global variables to input arguments by absl.FLAG
* Obstacle id for EV is not stable
* Calibration of respeaker direction -- which direction is 0 degree