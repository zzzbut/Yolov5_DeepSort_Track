
## 这是我的博客中关于目标跟踪的介绍，分别测试了三个代码，主要用现在这个：https://blog.csdn.net/qq_45874142/article/details/132454228
# 1.介绍
本文将介绍如何使用yolov5和deepsort进行目标检测和跟踪，并增加轨迹线的显示。本文的改进包括轨迹线颜色与目标框匹配、优化轨迹线只显示一段，并且当目标消失时不显示轨迹线。
# 2.效果展示
![man-min.gif](https://z4a.net/images/2023/08/25/man-min.gif)

# 3.如何使用
### （1）首先，我们需要下载本代码：
`git clone https://github.com/zzzbut/Yolov5_DeepSort_Track.git`

需要用到yolov5原预训练模型和行人重识别模型ckpt.t7，代码上传限制文件大小。可从下方的参考代码中获取，是个百度网盘链接。

### （2）接下来，我们需要修改track.py文件中的参数：
我们需要将参数修改为我们自己的视频文件路径和yolov5的预训练模型路径。我们还需要将编码方式设置为FLV1，这样保存的mp4文件才能打开。我们可以使用以下命令运行。

`python track.py --fourcc FLV1`

最后，我们可以看到目标检测和跟踪的结果，并且每个目标的轨迹线与其目标框匹配。当目标消失时，轨迹线也会消失，这样可以更清晰地看到每个目标的移动轨迹。
# 4.参考代码链接
十分感谢大佬的代码：
[Deepsort跟踪算法画目标运动轨迹](https://blog.csdn.net/qq_35832521/article/details/115124521?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522169269914116800222876736%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=169269914116800222876736&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-4-115124521-null-null.142%5Ev93%5EchatgptT3_2&utm_term=deepsort%20%E8%BD%A8%E8%BF%B9&spm=1018.2226.3001.4187)
