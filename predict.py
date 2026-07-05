from ultralytics import YOLO


def main():

    model = YOLO("best.pt")

    results = model.predict(
        source="images/test",
        imgsz=640,
        conf=0.25,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/detect",
        name="predict",
        exist_ok=True
    )

    print("Done!")


if __name__ == "__main__":
    main()