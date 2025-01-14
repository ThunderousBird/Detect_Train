from ultralytics import YOLO

def train_model():
    # 加载模型
    model = YOLO("yolov8x.yaml")  # YOLOv8 模型配置文件，可选 yolov8s.yaml 或 yolov8m.yaml

    # 训练模型
    model.train(
        data="dataset.yaml",  # 数据集配置文件路径
        epochs=50,            # 训练轮数
        imgsz=1920,            # 输入图像尺寸
        batch=16,             # 批量大小
        name="train_track_carriage",  # 项目名称
        device=0,             # 使用的 GPU ID，设置为 "cpu" 可在 CPU 上训练
        augment=True          # 启用数据增强
    )

    # 导出模型权重
    model.export(format="onnx")  # 导出为 ONNX 格式，支持多平台部署

if __name__ == "__main__":
    train_model()
