from ultralytics import YOLO


def main():

    model = YOLO("best.pt")

    metrics = model.val(
        data="dataset.yaml",
        imgsz=640,
        batch=8,
        split="test"   # или "val"
    )

    print(metrics)


if __name__ == "__main__":
    main()