# Emergency Vehicle Data Crawler

## Requirements

Additional dependencies you need to run this job:
```bash
pip3 install youtube-data-api youtube-dl
```

## Usage

To change the search content, download directory and the amount of data to crawl,
you'll need to change the run() function in emergency_vehicle_data_crawler.py

```bash
bazel run //fueling/perception/emergency_detection:emergency_vehicle_data_crawler
```

The emergency_vehicle_data_crawler requires youtube data api which is blocked by the
Firewall when submitting to kubernete cloud, so you can only run this crawler locally.
By default, the emergency vehicle videos will be downloaded and stored at:

	/apollo-fuel/fueling/perception/emergency_detection/data/<XXX>Vid

Images generated from videos will be stored at:

	/apollo-fuel/fueling/perception/emergency_detection/data/<XXX>Vid/image_clips

Audios will be downloaded and stored at:
	
	/apollo-fuel/fueling/perception/emergency_detection/data/<XXX>Aud

Once download is finished, every copy of files will be uploaded to bos, and a same file 
structure can be found here in bos:

	/mnt/bos/modules/perception/emergency_detection/data/emergency_vehicle/data
