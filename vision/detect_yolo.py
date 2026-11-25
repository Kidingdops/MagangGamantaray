"""
Sistem Deteksi Objek Terintegrasi - Gamantaray (v2 - Anti-Flicker)
Menghubungkan:
1. Kamera + AI YOLO (Bola Merah, Bola Hijau, Kotak Biru)
2. Aktuator LED GPIO (Merah, Hijau, Biru) - dengan State Persistence (Anti-Kedip)
3. Website Monitoring (Upload Otomatis Saat Kotak Biru Terdeteksi)
"""

import os
import sys
import time
import argparse
import threading
import cv2
import requests
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hardware.led_controller import LEDController

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model utama yang digunakan
MODEL_PATHS = [
    os.path.join(CURRENT_DIR, "buoys-redgreen_yolo11n.pt"),
    os.path.join(CURRENT_DIR, "kotak-biruhijau_yolo11n.pt"),
]

# Turunkan ambang batas deteksi agar kotak biru lebih stabil tertangkap
CONF_THRESHOLD = 0.35

# Jeda minimal antar upload foto ke website (detik)
UPLOAD_COOLDOWN_SEC = 3.0

# Waktu tahan (Hold Time) LED dalam detik agar LED tidak berkedip saat ada 1-2 frame lolos
LED_HOLD_TIME_SEC = 0.8


class ObjectDetectorAI:
    def __init__(self):
        self.models = []
        for path in MODEL_PATHS:
            if os.path.exists(path):
                print(f"[OK] Memuat Model: {os.path.basename(path)}")
                model = YOLO(path)
                self.models.append((os.path.basename(path), model))

        if not self.models:
            raise FileNotFoundError("Tidak ada file model .pt yang ditemukan di folder vision/!")

    def detect(self, frame):
        detected_candidates = []
        annotated_frame = frame.copy()

        for model_name, model in self.models:
            results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    raw_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2 - x1) * (y2 - y1)

                    raw_lower = raw_name.lower()

                    if "bola_merah" in raw_lower or "red" in raw_lower:
                        category = "RED"
                        display_name = "Bola Merah"
                        bgr_color = (0, 0, 255)
                    elif "bola_hijau" in raw_lower or ("green" in raw_lower and "box" not in raw_lower):
                        category = "GREEN"
                        display_name = "Bola Hijau"
                        bgr_color = (0, 255, 0)
                    elif "blue_box" in raw_lower or "kotak_biru" in raw_lower or "blue" in raw_lower:
                        category = "BLUE"
                        display_name = "Kotak Biru"
                        bgr_color = (255, 120, 0)
                    elif "green_box" in raw_lower:
                        category = "OTHER"
                        display_name = "Kotak Hijau"
                        bgr_color = (0, 180, 0)
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

        valid_targets = [c for c in detected_candidates if c["category"] in ("RED", "GREEN", "BLUE")]

        if valid_targets:
            valid_targets.sort(key=lambda item: item["area"], reverse=True)
            best = valid_targets[0]
            raw_action = best["category"]
            label_text = f"{best['display_name']} ({best['conf']:.2f})"
            banner_color = best["bgr_color"]
            best_box = best["box"]
        else:
            raw_action = "NONE"
            label_text = "Tidak Ada Objek"
            banner_color = (80, 80, 80)
            best_box = None

        return raw_action, label_text, annotated_frame, best_box, banner_color


def async_upload_photo(server_url, frame):
    def _upload():
        try:
            ret, jpeg_buffer = cv2.imencode(".jpg", frame)
            if not ret:
                return

            files = {"file": ("kotak_biru.jpg", jpeg_buffer.tobytes(), "image/jpeg")}
            url = f"{server_url.rstrip('/')}/upload"
            resp = requests.post(url, files=files, timeout=3.0)
            if resp.status_code == 200:
                print(f"[WEB UPLOAD] Foto Kotak Biru berhasil dikirim ke {url}")
        except Exception as e:
            print(f"[WEB UPLOAD ERROR] Gagal mengirim foto: {e}")

    threading.Thread(target=_upload, daemon=True).start()


def async_update_status(server_url, obj_name, led_state):
    def _send():
        try:
            url = f"{server_url.rstrip('/')}/api/status"
            requests.post(url, json={"object": obj_name, "led": led_state}, timeout=1.5)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Deteksi Objek Terintegrasi Gamantaray")
    parser.add_argument("--device", type=int, default=0, help="Index kamera (default: 0)")
    parser.add_argument("--server", type=str, default="http://localhost:5000", help="URL Web Server Monitoring")
    parser.add_argument("--no-gui", action="store_true", help="Jalankan mode headless / SSH")
    parser.add_argument("--conf", type=float, default=0.35, help="Ambang batas confidence (default: 0.35)")
    args = parser.parse_args()

    has_display = ("DISPLAY" in os.environ) or (sys.platform == "win32")
    show_gui = (not args.no_gui) and has_display

    print("==================================================")
    print("SISTEM INTEGRASI KENDALI & DETEKSI OBJEK GAMANTARAY")
    print(f"Kamera Index     : {args.device}")
    print(f"Server Monitoring: {args.server}")
    print(f"Ambang Conf      : {args.conf}")
    print(f"Mode Display     : {'GUI Aktif' if show_gui else 'Headless / Terminal'}")
    print("==================================================")

    global CONF_THRESHOLD
    CONF_THRESHOLD = args.conf

    detector = ObjectDetectorAI()
    led = LEDController()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"ERROR: Tidak dapat membuka kamera index {args.device}.")
        led.cleanup()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    active_led_state = "NONE"
    last_detected_action = "NONE"
    last_detect_time = 0
    last_upload_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            raw_action, label, annotated_frame, _, banner_color = detector.detect(frame)
            now = time.time()

            # ==========================================================
            # LOGIKA ANTI-KEDIP (State Persistence / Hold Filter)
            # ==========================================================
            if raw_action in ("RED", "GREEN", "BLUE"):
                # Objek terdeteksi pada frame ini
                active_led_state = raw_action
                last_detected_action = raw_action
                last_detect_time = now
            else:
                # Jika objek sempat luput 1-2 frame, jangan langsung matikan LED!
                # Tahan nyala LED selama LED_HOLD_TIME_SEC detik
                if (now - last_detect_time) > LED_HOLD_TIME_SEC:
                    active_led_state = "NONE"

            # 1. Update ke Hardware LED GPIO
            led.set_state(active_led_state)

            # 2. Update status ke terminal & website jika ada perubahan
            if raw_action != "NONE":
                status_display = label
            else:
                status_display = "Tidak Ada Objek" if active_led_state == "NONE" else f"{last_detected_action} (Holding)"

            # 3. Logika Khusus Kotak Biru: Ambil foto dan kirim ke website
            if raw_action == "BLUE" or active_led_state == "BLUE":
                if (now - last_upload_time) > UPLOAD_COOLDOWN_SEC:
                    print(f"[TRIGGER] Kotak Biru terdeteksi stabil! Mengirim foto ke website...")
                    async_upload_photo(args.server, annotated_frame)
                    last_upload_time = now

            # Tampilkan banner status
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 45), (30, 30, 30), -1)
            cv2.putText(
                annotated_frame,
                f"Status: {status_display} | LED: {active_led_state}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                banner_color,
                2
            )

            if show_gui:
                cv2.imshow("Gamantaray - Vision & Actuator", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh user.")
    finally:
        cap.release()
        led.cleanup()
        if show_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
