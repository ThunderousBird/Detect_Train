# 训练部分
## 数据预处理
1. 视频处理：将视频装换到统一的格式（1920*1080）。
2. 视频帧提取：从视频中逐帧提取图像，确保帧率适合后续分析，这里采用100帧截取一张图片，方式图片相似度较高。
3. 数据标注：为每帧标注车厢和轨道的位置及类别（车厢类标签如 carriage）；轨道标签采用一张没有列车出现的空轨道图片进行标注（track_1, track_2）等。
4. 数据增强：通过旋转、翻转、噪声添加等技术增加数据多样性，提升模型鲁棒性。
## 数据集制作：
YOLOv8 Segmentation格式。每张图像的标注存储在 .txt 文件中，包含物体类别和边界框（class x_center y_center width height）。使用 .txt 文件保存铁轨的多边形坐标。
## 模型训练
模型：YOLOv8。工具：Ultralytics YOLO 开源工具。流程：数据加载：加载预处理后的数据集。模型配置：设定类别数、输入尺寸、超参数（学习率、批量大小等）。开始训练：多次迭代优化模型权重。验证与测试：使用验证集评估性能指标（如 mAP）。
# 推理部分
## 整体设计流程
1. 输入数据：视频的某一帧图像数据。
2. 数据预处理：将输入图像缩放到模型的标准尺寸（如 1920*1080）。
3. 模型运行（轨道）：首先选取不存在列车的清晰图片进行轨道区域的识别，将得到的信息通过opencv内置函数进行边框提取，并将信息保存为.npy格式在./contours文件夹下。
4. 模型运行（识别）：由于摄像头的位置和角度较为固定，因此利用每条轨道的边框数据可以进行对比。使用 YOLO 模型检测图像中的车厢位置，取置信度大于0.7的标签（防止误判和较远距离的火车影响）。将检测结果映射到每条轨道的状态。对两个边框进行对比，如果重合度大于轨道边框的 80% 则认为火车在轨。
5. 输出数据：格式：JSON 文件，记录每帧画面中各轨道的状态（"occupied"/"empty"）。
## 流程说明
输入数据（处理后的视频某一帧图像数据）->数据预处理（保证1920*1080大小）->模型运行（推理）->输出数据（结果输出为json文件track_status.json，保存在./output_json，识别图片保存在./output_pic）