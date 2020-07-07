<!--
 * @Author: xieydd
 * @since: 2020-05-22 00:47:23
 * @lastTime: 2020-05-22 19:15:15
 * @LastAuthor: Do not edit
 * @message: 
--> 
### Test Summary

|  Model   | Easy  | Medium | Hard | FLOPs(M)
|  ----  | ----  | ---- | ---- | ----
| MTCNN  | 0.851 | 0.820 | 0.607 |
| RetinaFace(mobilenet-0.25)(640x480) | 0.873 | 0.836 | 0.638
| 1M-RFB(640x480) | 0.865 | 0.828| 0.622 
| 1M-slim(640x480) | 0.850	| 0.808	| 0.595
| RetinaFace(mobilenet-0.25)(320x240) | 0.811	| 0.697	 |0.376	 |193.921
| 1M-RFB(320x240)	|0.814	|0.710	|0.363	|118.435
| 1M-slim(320x240)	|0.795	|0.683	|0.345	|98.793
|1M-slim(256)	|0.751	|0.609	|0.278	|59.232
|lightface-1(320x240)	|0.736	|0.564	|0.250	|65.607
|lightface-1(256)	|0.655	|0.446	|0.189	|41.426
|lightface-2(320x240)	|0.738	|0.564	|0.251	|85.552
|lightface-1(640x480)	|0.828	|0.775	|0.476	|256.924
|lightface-2(640x480)	|0.824	|0.778	|0.476	|335.336
|lightface-3(320x240)	|0.803	|0.700	|0.350	|95.585
|lightface-3(256)	|0.764	|0.624	|0.282	|60.624
|lightface-3(640x480)	|0.858	|0.814	|0.595	|379.541
| CenterFace Paper| 0.935|0.924|0.875|
|CenterFace（mobilenetv2 0.25）	300x300 |	0.617|	0.598|	0.363|	288
| libfacedetection(origin scale)|	0.835|	0.806|	0.626|	 
|libfacedetection(320)|	0.801|	0.666|	0.297|	264.947
| Advanced VGG 640|0.905|0.889|0.765|10G
|EfficientDet 640|0.876|0.846|0.682|
|EfficientNet b3 640|0.906 |    0.866   |   0.707| 360M
|EfficientNet b4 640|0.875 |    0.845   |   0.696| 436M
|EfficientNet b3 640 + FPN+SSH context module|0.927 |    0.872   |   0.671| 2.46G
|EfficientNet b3 640 + FPN+SSH context module + margin based classification loss|0.923 |    0.887   |   0.714| 2.46G

2020-05-16
结论： 
1. EfficientNet b3 640 + FPN+SSH context module + margin based classification loss margin 中 m 为0.1 fpn 采用 scale_factor, 640 输入下为 0.923 0.893 0.720 作为 CPU Server 版本
2. 由于 efficient系列模型在 openvino reshape 不了，改用改进版 VGG+SHH+margined based classification loss , FLOPs 2.7G val指标 0.880 0.848 0.673 关键点好于1


- lightface1: 采用 1M Slim 那边的backbone，改进版shufflenet v2 0.5，同时将原有的anchor数10提高到13，FLOPs为 65.607M
- lightface2: 参考 [find-tiny-face](https://github.com/varunagrawal/tiny-faces-pytorch) 将各层特征进行一个融合，提高小人脸检出率，FLOPs为 85.552M
- lightface3: 直接在slim基础上，将anchor数10提高到13

#### Test
1. VGG_Light 320
Easy   Val AP: 0.8089601441130336
Medium Val AP: 0.6488957359945599
Hard   Val AP: 0.3033894512658072
2. VGG_Light 640
Easy   Val AP: 0.890395149819784
Medium Val AP: 0.8357327675807253
Hard   Val AP: 0.5530419590609832