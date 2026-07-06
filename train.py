from ultralytics import YOLO


def main():

    model = YOLO("yolov5n.pt")  # можно оставить v5 веса

    model.train(
        data="dataset.yaml",
        epochs=40,
        imgsz=640,
        batch=8,
        device=0,
        project="runs",
        name="airspace_detector",
        exist_ok=True
    )


if __name__ == "__main__":
    main()