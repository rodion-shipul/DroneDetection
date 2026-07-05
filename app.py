import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO


MODEL_PATH = "runs/drone_detector/weights/best.pt"


class DroneDetectionApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Обнаружение БПЛА (Ultralytics YOLO)")
        self.root.geometry("1000x700")

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror(
                "Ошибка",
                f"Не найдена модель:\n\n{MODEL_PATH}\n\nСначала обучи модель."
            )
            self.root.destroy()
            return

        # ✅ НОВЫЙ ПРАВИЛЬНЫЙ ЗАГРУЗЧИК
        self.model = YOLO(MODEL_PATH)

        # UI
        self.title_label = tk.Label(
            root,
            text="Система обнаружения БПЛА (YOLO)",
            font=("Arial", 18, "bold")
        )
        self.title_label.pack(pady=10)

        self.open_button = tk.Button(
            root,
            text="Загрузить изображение",
            font=("Arial", 14),
            command=self.open_image
        )
        self.open_button.pack(pady=10)

        self.result_label = tk.Label(
            root,
            text="Выберите изображение для анализа",
            font=("Arial", 12),
            justify="left"
        )
        self.result_label.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def open_image(self):

        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            # 🔥 inference (Ultralytics)
            results = self.model.predict(source=file_path, conf=0.25)

            result = results[0]

            # annotated image
            img = result.plot()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(img)
            img.thumbnail((900, 500))

            photo = ImageTk.PhotoImage(img)

            self.image_label.configure(image=photo)
            self.image_label.image = photo

            # results text
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                self.result_label.config(text="БПЛА не обнаружены.")
                return

            output = f"Обнаружено объектов: {len(boxes)}\n\n"

            for i in range(len(boxes)):

                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                name = self.model.names[cls_id]

                output += (
                    f"{i+1}. {name}\n"
                    f"   Вероятность: {conf:.2%}\n\n"
                )

            self.result_label.config(text=output)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))


if __name__ == "__main__":

    root = tk.Tk()
    app = DroneDetectionApp(root)
    root.mainloop()