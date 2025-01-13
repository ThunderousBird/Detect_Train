import cv2
import json
import logging
from ultralytics import YOLO

# logging.getLogger("ultralytics").setLevel(logging.WARNING)
def detect_tracks(video_path, output_path):
    """
    使用 YOLO 模型检测视频中每条铁轨是否有车厢存在。
    
    :param video_path: 输入视频文件路径
    :param output_path: 输出 JSON 文件路径
    """
    # 加载预训练的 YOLOv8 模型
    model = YOLO('yolov8n.pt')  # 使用轻量级 YOLOv8 Nano 预训练模型

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    results = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        # YOLO 推理
        detections = model(frame)
        frame_status = {"frame_id": frame_id, "track_status": {}}

        # 初始化轨道状态
        tracks = {"track_1": "empty", "track_2": "empty", "track_3": "empty"}

        # 解析检测结果
        for detection in detections[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = detection
            class_id = int(class_id)
            label = model.names[class_id]

            # 假设模型类别名包含 "track" 相关名称
            if "track" in label:
                track_id = label.split("_")[1]  # 假设类别名如 "track_1"
                tracks[track_id] = "occupied"

        frame_status["track_status"] = tracks
        results.append(frame_status)

    cap.release()

    # 将检测结果保存为 JSON 文件
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detection results saved to {output_path}")

if __name__ == '__main__':
    # 修改这里的路径为您的文件路径
    video_path = "./part-test.mp4"  # 输入视频文件路径
    output_path = "./output.json"  # 输出 JSON 文件路径

    # 执行检测
    detect_tracks(video_path, output_path)
