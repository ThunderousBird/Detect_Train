import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

def load_and_fix_track_contours(track_file_path):
    """
    加载轨道轮廓并修复格式问题，以适配 OpenCV 需要的格式。
    
    :param track_file_path: .npy 文件路径
    :return: 修复后的轮廓列表
    """
    try:
        # 加载数据
        raw_contours = np.load(track_file_path, allow_pickle=True)

        # 修复格式
        fixed_contours = []
        for contour in raw_contours:
            # 如果 contour 是有效的 NumPy 数组
            if isinstance(contour, np.ndarray):
                # 将 contour 转换为 (N, 1, 2) 格式
                contour = np.array(contour, dtype=np.int32)
                if contour.ndim == 2:
                    contour = contour[:, np.newaxis, :]
                elif contour.ndim != 3 or contour.shape[2] != 2:
                    print(f"Skipping invalid contour: {contour}")
                    continue
                fixed_contours.append(contour)
            else:
                print(f"Skipping invalid contour: {contour}")

        return fixed_contours
    except Exception as e:
        print(f"Error loading or fixing track contours from {track_file_path}: {e}")
        return None


def is_box_overlapping_contour(bbox, contour):
    """
    检测边界框是否与轨道轮廓重叠。
    
    :param bbox: 边界框 (x1, y1, x2, y2)
    :param contour: 单个轨道轮廓 (必须为 (N, 1, 2) 格式)
    :return: True 如果重叠，否则 False
    """
    x1, y1, x2, y2 = bbox

    # 检查边界框的四个角点是否在轮廓内
    bbox_points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for point in bbox_points:
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            return True

    return False


def detect_and_draw(image_path, output_image_path, track_files):
    """
    使用 YOLO 模型检测图片中的火车，并分别判断是否与轨道重叠。
    :param image_path: 输入图片文件路径
    :param output_image_path: 输出图片文件路径
    :param track_files: 轨道轮廓文件路径列表，每个文件对应一个轨道
    """

    # 加载预训练的 YOLO 模型
    model = YOLO('model/yolov8x.pt')

    # 需要的类别和置信度阈值
    TARGET_CLASSES = {"train"}  # 替换为你的目标类别
    CONFIDENCE_THRESHOLD = 0.3

    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot open image file {image_path}")
        return

    # 加载轨道轮廓
    contours = {}
    for i, track_file in enumerate(track_files):
        contour = load_and_fix_track_contours(track_file)
        if contour is not None:
            contours[f"Track_{i+1}"] = contour

    # 使用 YOLO 模型进行推理
    results = model(image)

    # 处理检测结果
    for detection in results[0].boxes.data.tolist():
        x1, y1, x2, y2, conf, class_id = detection
        class_id = int(class_id)
        label = model.names[class_id]

        # 置信度过滤
        if label in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:
            # 默认设置边框颜色为红色（不在轨道上）
            color = (0, 0, 255)
            text = f"{label} {conf:.2f} Off Track"
            bbox = (x1, y1, x2, y2)

            # 遍历每个轨道，检查是否有重叠
            on_track = False
            for track_name, track_contours in contours.items():
                for contour in track_contours:  # 遍历轨道的每个分段
                    if is_box_overlapping_contour(bbox, contour):
                        color = (0, 255, 0)  # 如果与某轨道重叠，改为绿色
                        text = f"{label} {conf:.2f} On {track_name}"  # 包含轨道名称
                        print(f"Train detected on {track_name}")
                        on_track = True
                        break
                if on_track:
                    break  # 已经确认在轨道上，退出循环

            if not on_track:
                print("Train detected off all tracks.")

            # 在图片上画出边界框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    track_files = [
        "contours/track1_contours.npy",
        "contours/track2_contours.npy",
        "contours/track3_contours.npy",
        "contours/track4_contours.npy",
        "contours/track5_contours.npy",
        "contours/track6_contours.npy",
        "contours/track7_contours.npy",
        "contours/track8_contours.npy",
    ]
    image = draw_tracks_on_image(image, track_files, output_image_path)
    # 保存带检测框的图片
    cv2.imwrite(output_image_path, image)
    print(f"Result image saved to {output_image_path}")

def draw_tracks_on_image(image, track_files, output_image_path):
    """
    在指定图片上绘制轨道轮廓。
    
    :param image_path: 输入图片路径
    :param track_files: 轨道轮廓文件路径列表
    :param output_image_path: 输出图片路径
    """
    # 读取原始图片
    # image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot open image file")
        return

    for i, track_file in enumerate(track_files):
        track_name = f"Track_{i + 1}"
        track_contours = load_and_fix_track_contours(track_file)

        if track_contours is not None:
            for contour in track_contours:
                # 绘制轨道轮廓
                cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)  # 蓝色表示轨道

                # 计算质心并标记轨道名称
                moments = cv2.moments(contour)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cv2.putText(image, track_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # else:
                    # print(f"Track {track_name}: Contour has zero area, skipping text placement.")
        else:
            print(f"Skipping invalid or empty track file: {track_file}")

    # # 保存并显示图片
    cv2.imwrite(output_image_path, image)
    print(f"Tracks drawn and saved to {output_image_path}")

    # 显示图片（可选）
    # cv2.imshow("Tracks", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

if __name__ == '__main__':
    # 输入图片路径
    input_folder = "data"
    # 输出图片路径
    output_folder = "output_pic"

    # 轨道坐标文件路径列表
    track_files = [
        "contours/track1_contours.npy",
        "contours/track2_contours.npy",
        "contours/track3_contours.npy",
        "contours/track4_contours.npy",
        "contours/track5_contours.npy",
        "contours/track6_contours.npy",
        "contours/track7_contours.npy",
        "contours/track8_contours.npy",
    ]

    # 执行检测并画出框
    # detect_and_draw(image_path, output_image_path, track_files)
    for filename in os.listdir(input_folder):
        # 检查文件是否为图片
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构建完整输入路径和输出路径
            image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            # 执行检测并保存结果
            print(f"Processing: {image_path}")
            detect_and_draw(image_path, output_image_path, track_files)
