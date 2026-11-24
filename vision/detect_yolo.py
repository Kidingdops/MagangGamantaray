"""
Modul Deteksi Objek berbasis AI (YOLO)
Sistem Kendali & Deteksi Objek Gamantaray

Mendukung model:
- buoys-redgreen_yolo11n.pt (bola_merah, bola_hijau)
- buoys-blue_yolo11n.pt / kotak_biru (blue_buoy / kotak_biru)
"""

import os
import sys
import time
import cv2
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# File model yang baru saja kamu sediakan
MODEL_REDGREEN_PATH = os.path.join(CURRENT_DIR, "buoys-redgreen_yolo11n.pt")
MODEL_BLUE_PATH = os.path.join(CURRENT_DIR, "buoys-blue_yolo11n.pt")

CONF_THRESHOLD = 0.45


class ObjectDetectorAI:
    def __init__(self):
        self.models = []
        
        # Muat model bola merah & hijau
        if os.path.exists(MODEL_REDGREEN_PATH):
            print(f"[OK] Memuat Model Bola Merah/Hijau: {MODEL_REDGREEN_PATH}")
            self.models.append(YOLO(MODEL_REDGREEN_PATH))
        else:
            print(f"[WARN] File {MODEL_REDGREEN_PATH} tidak ditemukan.")

        # Muat model biru
        if os.path.exists(MODEL_BLUE_PATH):
            print(f"[OK] Memuat Model Objek Biru: {MODEL_BLUE_PATH}")
            self.models.append(YOLO(MODEL_BLUE_PATH))
        else:
            print(f"[WARN] File {MODEL_BLUE_PATH} tidak ditemukan.")

        if not self.models:
            raise FileNotFoundError("Tidak ada file model .pt yang ditemukan di folder vision/!")

    def detect(self, frame):
        """
        Menjalankan deteksi pada frame.
        Mengembalikan:
        - target_action: 'RED', 'GREEN', 'BLUE', atau 'NONE'
        - label_text: nama objek lengkap
        - annotated_frame: frame dengan kotak dan label
        - best_box: koordinat kotak (x1, y1, x2, y2)
        """
        detected_candidates = []
        annotated_frame = frame.copy()

        for model in self.models:
            results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    raw_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2 - x1) * (y2 - y1)

                    # Standarisasi label & aksi sesuai dokumen tugas
                    raw_lower = raw_name.lower()
                    if "merah" in raw_lower or "red" in raw_lower:
                        category = "RED"
                        display_name = "Bola Merah"
                        bgr_color = (0, 0, 255)
                    elif "hijau" in raw_lower or "green" in raw_lower:
                        category = "GREEN"
                        display_name = "Bola Hijau"
                        bgr_color = (0, 255, 0)
                    elif "blue" in raw_lower or "biru" in raw_lower or "kotak" in raw_lower:
                        category = "BLUE"
                        display_name = "Kotak Biru" if "kotak" in raw_lower else "Objek Biru"
                        bgr_color = (255, 120, 0)
                    else:
                        category = "UNKNOWN"
                        display_name = raw_name
                        bgr_color = (200, 200, 200)

                    detected_candidates.append({
                        "category": category,
                        "display_name": display_name,
                        "conf": conf,
                        "box": (x1, y1, x2, y2),
                        "area": area,
                        "bgr_color": bgr_color
                    })

                    # Gambar kotak deteksi
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), bgr_color, 2)
                    cv2.putText(
                        annotated_frame,
                        f"{display_name} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        bgr_color,
                        2
                    )

        # Jika ada objek yang terdeteksi, pilih objek terbesar
        if detected_candidates:
            detected_candidates.sort(key=lambda item: item["area"], reverse=True)
            best = detected_candidates[0]
            target_action = best["category"]
            label_text = f"{best['display_name']} ({best['conf']:.2f})"
            banner_color = best["bgr_color"]
            best_box = best["box"]
        else:
            target_action = "NONE"
            label_text = "Tidak Ada Objek"
            banner_color = (80, 80, 80)
            best_box = None

        # Banner status di bagian atas layar
        cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 45), (30, 30, 30), -1)
        cv2.putText(
            annotated_frame,
            f"Status: {label_text} | Aksi: LED {target_action}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            banner_color,
            2
        )

        return target_action, label_text, annotated_frame, best_box


def main():
    print("==================================================")
    print("Memulai Deteksi AI Gamantaray (YOLO11)")
    print("Tekan 'q' di jendela kamera untuk keluar.")
    print("==================================================")

    detector = ObjectDetectorAI()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Tidak dapat mengakses kamera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_time = time.time()
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Hitung FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - fps_time) if curr_time > fps_time else 0
        fps_time = curr_time

        action, label, annotated_frame, _ = detector.detect(frame)

        # Tampilkan FPS di pojok kanan atas
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (annotated_frame.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("Gamantaray - Deteksi Objek AI", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
