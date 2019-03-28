# Raw Data Frame Filtering Rules For Data Labeling

## 2019-03-28

### General
1. Collect data with following sensors open and keep desired frame rates respectively
    - Lidar 128 (/apollo/sensor/lidar128/compensator/PointCloud2), frame rate: 10
    - Front camera 6mm (/apollo/sensor/camera/rear_6mm/image/compressed), frame rate: 10+
    - Front camera 12mm (/apollo/sensor/camera/front_12mm/image/compressed), frame rate: 10+
    - Left fisheye camera (/apollo/sensor/camera/left_fisheye/image/compressed), frame rate: 10+
    - Right fisheye camera (/apollo/sensor/camera/right_fisheye/image/compressed), frame rate: 10+
    - Front Radar (/apollo/sensor/radar/front), frame rate: 2+
    - Rear Radar (/apollo/sensor/radar/rear), frame rate: 2+
    - Localization pose (/apollo/localization/pose), frame rate: 90+

2. Make sure Localization information is accurate.  Favor MSF localization mode over RTK mode.

3. Upload data to NFS or BOS after data collection

4. Extract above sensors data from recorded files, and do the following matches:
    - Match <font color=#0000ff>*every*</font> Lidar frame with <font color=#0000ff>*all*</font> the <font color=#0000ff>*closest*</font> other sensors frame
    - Match <font color=#0000ff>*every*</font> sensor frame with <font color=#0000ff>*closest*</font> pose frame
   <br>For example, for each lidar frame we should find all the camera frames that have smallest time diff with it, 
   <br>and meanwhile, find closest pose frames to the cameras frames as well.
  
5. Generate desired format of output corresponding to different clients requirements and serialize to disk.

6. Send over the frames for labeling.

7. The above extraction and serialization must be reliable and efficient.  Targeting throughput is <font color=#0000ff>*10K frames*</font> per day.

### Special Requirements for Scale
1. Provide GPSLocation information, including the heading and position, for vehicle itself and devices, including Lidar, Camera, Radar, and Pose.

2. All the coordinates should be in world-coordinate system.

3. Organize the sensors data in Json format files by following [the document](https://private-docs.scale.ai/#data-types-and-the-frame-objects), and send the tasks over to Scale by issuing requests.

4. Tasks must be sent batch by batch, each batch should be exactly 50 frames

5. Each batch should start from a *stationary pole frame*, means starting point to which all other frames' locations are relative.

6. Filter out frames with time stamp difference between front 6mm camera and 128 lidar  >  30ms

7. Extract 2 frames per second on average from the remaining frames after the "30ms time difference" rule

8. Use GPS information to filter out frames with movement less than 1 meter relative to the previous frame, after the "2 frames per second" rule

9. Ask Scale to label 2D box in the front 6mm camera and all 3D boxes on lidar frame.

10. Targeting to provide 20K frames this quarter.  Means we need ~200K raw data after applying all filter rules.
 
### Special Requirements for Beijing
1. Provide Lidar 128, Pictures of 5 cameras, and Pose information.

2. Lidar 128 pointcloud should be in *compressed binary* format.  

3. All frames including Lidar and Camera should be in continuous sequence order, and indexing text files are needed to indicate the mapping between Lidar and Camera, Lidar and Pose, Lidar and timestamp.

4. Tasks should be submitted batch by batch, frames inside of each batch should be continuous.

5. Filter out frames with time stamp difference between front 6mm camera and 128 lidar  >  100ms

6. Use original frame numbers (before 100ms time difference rule) to extract 1 frame for every 3 raw frames. If the frame correspond to the extracted frame number was deleted after 100ms time difference rule, this track is considered as completed and start the next new track with the next available extracted frame.

7. Ask Beijing to label 2d box in the front 6mm camera and all 3D boxes on lidar frame, but these 3D boxes are 'tight' bbox, only include points visible in the lidar frame. And link 2D-3D objects. Also track objects visible in the front 6mm camera. 

8. Targeting to provide 100K frames this quarter.  Means we need ~400K raw data after applying all filter rules.
