Learning Wheel Odometry and IMU Errors module introduced a methodology for Gaussian processes with neural networks and stochastic variational inference to correct the trajectory errors of localization, by learning error residuals between physical prediction and ground truth data. These corrections are also shown to be usable for EKF design

<!-- # Table of Contents 1\. [Frontend](#frontend) 2\. [Data](#data) - [Upload Tool](#upload) - [Download Tool](#download) 3\. [Visulization](#visulization) -->

 # Running Codes under Local Conda Environment (Outside Docker)

## Environment Setup

Before running the code package, a series of preparation work need to be conducted as follows:

1. Set up the correct Conda environment:

      ```bash
      $ conda env update --prune -f /apollo/modules/data/fuel/conda/py36-pyro.yaml
      $ conda activate fuel-py36-pyro.yaml
      ```
    Note: the `fuel-py36-pyro` is a little different from the regualr `fuel-py36` environment. The former one includes some special version setting which matches the current `pyro-ppl` version.     

## Download Sensor and Groundtruth Data

So far the lwoi_localization package only supports the NCLT data downloading and training, due to the copyright agreement of Kaist data. The steps to download and settle the NCLT data package are shown as follows:

1. Download the NCLT sensor and groundtruth data with the downloader.py:

      ```bash
      $ python /apollo/modules/data/fuel/fueling/control/dynamic_model/lwoi_localization/downloader.py --sen
      $ python /apollo/modules/data/fuel/fueling/control/dynamic_model/lwoi_localization/downloader.py --gt
      ```
      The compressed data package will be download into the current directory

2. Settle the decompressed data into the corresponding paths under /apollo-fuel/testdata/:

   For example, the sensor_data and groundtruth data can be feed into the following folders:  

   ```bash
   .../apollo-fuel/testdata/control/lwoi_localization/sensor_data/nclt/training/2012-01-08/(data)
   .../apollo-fuel/testdata/control/lwoi_localization/sensor_data/nclt/training/2012-01-15/(data)
   .../apollo-fuel/testdata/control/lwoi_localization/sensor_data/nclt/training/2012-01-22/(data)
   .../apollo-fuel/testdata/control/lwoi_localization/sensor_data/nclt/cross_validation/2012-10-28/(data)
   .../apollo-fuel/testdata/control/lwoi_localization/sensor_data/nclt/test/2012-12-01/(data)

   .../apollo-fuel/testdata/control/lwoi_localization/ground_truth/nclt/groundtruth_2012-01-08.csv
   .../apollo-fuel/testdata/control/lwoi_localization/ground_truth/nclt/groundtruth_2012-01-15.csv
   .../apollo-fuel/testdata/control/lwoi_localization/ground_truth/nclt/groundtruth_2012-01-22.csv
   .../apollo-fuel/testdata/control/lwoi_localization/ground_truth/nclt/groundtruth_2012-10-28.csv
   .../apollo-fuel/testdata/control/lwoi_localization/ground_truth/nclt/groundtruth_2012-12-01.csv
   ```

   Note:

   (1) Sensor data `2012-05-26` and `2012-06-15` seems to be conflicted with some settings in the current codes. Temporarily it is suggested to avoid to use them;

   (2) The default data assignment in the literature is:
      - Training data: first 19 sequences (temporarily remove `2012-05-26` and `2012-06-15`)
      - Cross-validation data: `2012-10-28, 2012-11-04, 2012-11-16, 2012-11-17`
      - Testing data: `2012-12-01, 2013-01-10, 2013-02-23, 2013-04-05`

   (3) Before executing codes, please remove the nonsense files such as
      - `testdata/control/lwoi_localization/data/nclt/BLANK.txt`


## Run the trainning and test codes

The Main Codes are included in the lwoi_main.py.

1. Simply run the codes to process data, learning model, test resuls :

      ```bash
      $ python /apollo/modules/data/fuel/fueling/control/dynamic_model/lwoi_localization/main_lwoi.py
      ```

2. Advanced setting for input arguments (refer to the lwoi_main.py):

  ```bash
  '--nclt'             (default=True)
  '--path_data_base'   (default=testdata/control/lwoi_localization/)
  '--path_data_save'   (default=testdata/control/lwoi_localization/data/)
  '--path_results'     (default=testdata/control/lwoi_localization/results/)
  '--path_temp'        (default=testdata/control/lwoi_localization/temp/)
  ....... (All the data extraction, model parameters and optimizer parameters settings)
  ```

3. After running the codes,

   the generated processed dataset will be saved at `testdata/control/lwoi_localization/data/`;

   the generated model parameters files will be saved at `testdata/control/lwoi_localization/temp/`;

   the test m-ATE results will be real-time displayed on screen



 # Running Codes with Docker Conda Environment

 ## Environment Setup

 Before running the code package, a series of preparation work need to be conducted as follows:

 1. Set up the correct Conda environment:

       ```bash
       $ conda env update --prune -f /apollo/modules/data/fuel/fueling/conda/py36-pyro.yaml
       $ conda activate fuel-py36-pyro.yaml
       ```
## Download Sensor and Groundtruth Data

(The same as out-of-docker situation)

## Run the trainning and test codes

(The same as out-of-docker situation)

## Note: Possible code dump or errors happened inside docker

1. Core Dump due to Nvidia-docker issue:

   During running inside-docker, if a core dump emerges with information like

    `Segmentation fault (core dumped)`

    , then please first check whether the Nvidia-Docker2 is well installed outside the docker.
    The detailed installation guidance please follow:

   https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0)

2. Core Dump due to version incompatibility:

   The usage of the fuel-py36-pyro.yaml (instead of fuel-py36.yaml) is due to the incompatibility between the by-default pytorch version (`pytorch==1.1.0`, currently) and the stable version of pyro tool (`pyro-ppl==0.3.3`). This incompatibility will lead to some core-dump with information

   `Illegal instruction (core dumped)`

   when the pyro.infer module is called. Thus, in fuel-py36-pyro.yaml, `torch==1.1.0` (which is not equivalent to `pytorch=1.10`) is uploaded.  

3. Error Message:
   During running inside-docker, if a error message emerges similar to

   `.../miniconda2/envs/fuel-py36/lib/python3.6/site-packages/torch/lib/libtorch.so.1: undefined symbol:_ZN5torch27GetEmptyStringAlreadyInitedB5cxx11Ev`

   or, sometimes if a core dump emerges only with information

    `Segmentation fault (core dumped)`

   , then please run the following code to solve the problem:  

         $ export LD_LIBRARY_PATH=/usr/local/cuda/lib64

   here, `cuda` may need to be replaced by `cuda-9.0` or `cuda-10.0`, depends on the specific directory in your docker environment

4. Error Message:
   During running inside-docker, if some error message emerges similar to:

   `ERROR: ld.so: object 'libcaffe2_gpu.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.
   ERROR: ld.so: object 'libcaffe2_gpu.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.`

   Please just ignore them temporarily since they will not interfere with the operation of the codes.
