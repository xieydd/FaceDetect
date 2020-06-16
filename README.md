<!--
 * @Author: your name
 * @Date: 2020-05-23 21:58:08
 * @LastEditTime: 2020-06-04 21:07:16
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /FaceDetect/README.md
--> 
### Face Detection Summary



#### Training
```
# Change Training param in data/.config
$ python train.py --resume_net=./weights/slim_Final.pth --network=slim --resume_epoch=10
```

#### Calculate flops and model param size
```
$ cd utils && python calculate_paremeter_flop.py
```

#### Site
- [DBFace for single detector for face with more tricks](https://github.com/dlunion/DBFace)
- [1MB Face Detector with 5 landmark](https://github.com/biubug6/Face-Detector-1MB-with-landmark)
- [Faster and Generic 1MB Face Detector with more inference framework test](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/README_CN.md)
- [RetinaFace single Stage Face Detector](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- [CenterFace single stage and anchor free inprove generalization](https://github.com/Star-Clouds/CenterFace)
- [libfacedetection include high profermance inference with SIMD(intel avx2 and arm neon)](https://github.com/ShiqiYu/libfacedetection)
- [IoU loss for better bindingbox regression and cluster nms for enhancement object detection and instance segmetion](https://github.com/Zzh-tju/DIoU-SSD-pytorch)
