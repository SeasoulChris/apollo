

Offline training
Command:

    ```
bazel run emergency_detection_train_pipeline -- --cloud --gpu=1 --gpu_id=0 --classes=2 --learning_rate=0.001 --optimizer=adam --iou_type=iou --pretrained=/mnt/bos/modules/perception/emergency_detection/pretrained_model/yolov4.conv.137.pth --image_dir=/mnt/bos/modules/perception/emergency_detection/data/emergency_vehicle/images --label_dir=/mnt/bos/modules/perception/emergency_detection/data/emergency_vehicle --checkpoint_dir=/mnt/bos/modules/perception/emergency_detection/checkpoints --training_log_dir=/mnt/bos/modules/perception/emergency_detection/logs
    
    ```

Download the log at /mnt/bos/modules/perception/emergency_detection/log, use tensorboard to visualize the training loss and accuracy.




TensorRT and onnx version mapping
pytorch 1.5.1(current version), 1.6 -> Tensorrt 7.1.2 or higher (verified on 7.1.3.4)-> onnx 1.6, onnx ir version 0.0.6, cuda 10.2, cudnn 8.

Notice libtorch version, current version may confilict with cudnn 8.


pytorch 1.3, 1.4 -> Tensorrt 7.0.0(current version) or higher -> onnx 1.4, onnx ir version 0.0.4, cuda 10.1, cudnn 7.

