"""
Sistem Deteksi Objek Terintegrasi Berbasis OpenCV - Gamantaray
Menggunakan deteksi warna HSV & kontur untuk mendeteksi:
1. Bola Merah -> Aktuator LED Merah (GPIO 17)
2. Bola Hijau -> Aktuator LED Hijau (GPIO 27)
3. Kotak Biru -> Aktuator LED Biru (GPIO 22) + Upload Foto ke Website Monitoring

Dilengkapi:
- Logika Anti-Kedip (State Persistence / Hold Filter)
- Sinkronisasi Status ke Server Web Monitoring (/api/status)
- Upload Otomatis Foto Kotak Biru (/upload)
"""

import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np
import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hardware.led_controller import LEDController

# ==========================================================
# 1. RENTANG WARNA HSV (Bisa disesuaikan via calibrate_hsv.py)
# ==========================================================
# Warna Merah: Berada di kedua ujung spektrum Hue OpenCV (0-10 & 165-180)
LOWER_RED1 = np.array([0, 100, 60])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([165, 100, 60])
UPPER_RED2 = np.array([180, 255, 255])

# Warna Hijau: Rentang Hue sekitar 35 - 85
LOWER_GREEN = np.array([35, 70, 50])
UPPER_GREEN = np.array([85, 255, 255])

# Warna Biru: Rentang Hue sekitar 95 - 135
LOWER_BLUE = np.array([95, 100, 50])
UPPER_BLUE = np.array([135, 255, 255])

# Ambang batas luas kontur minimum (agar noise kecil tidak dianggap objek)
DEFAULT_MIN_CONTOUR_AREA = 1000

# Jeda minimal antar upload foto ke website (detik)
UPLOAD_COOLDOWN_SEC = 3.0

# Waktu tahan (Hold Time) LED dalam detik agar LED tidak berkedip saat ada 1-2 frame lolos
LED_HOLD_TIME_SEC = 0.8


class ColorDetectorOpenCV:
    """
    Detektor objek berbasis segmentasi warna HSV dan analisis kontur OpenCV.
    """
    def __init__(self, min_area=DEFAULT_MIN_CONTOUR_AREA):
        self.min_area = min_area

    def detect(self, frame):
        """
        Menganalisis frame kamera dan mengembalikan:
        (raw_action, label_text, annotated_frame, best_box, banner_color)
        - raw_action: 'RED' | 'GREEN' | 'BLUE' | 'NONE'
        - label_text: String deskripsi status
        - annotated_frame: Frame dengan visualisasi bounding box & info
        - best_box: (x1, y1, x2, y2) dari objek dominan atau None
        - banner_color: BGR tuple untuk warna banner
        """
        annotated_frame = frame.copy()

        # 1. Reduksi noise dengan Gaussian Blur
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 2. Buat Masking untuk masing-masing warna
        mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        # 3. Operasi morfologi (Open untuk buang noise, Close untuk menutup lubang kecil)
        kernel = np.ones((5, 5), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

        color_masks = [
            ("RED", "Bola Merah", mask_red, (0, 0, 255)),
            ("GREEN", "Bola Hijau", mask_green, (0, 255, 0)),
            ("BLUE", "Kotak Biru", mask_blue, (255, 120, 0)),
        ]

        detected_candidates = []

        # 4. Cari kontur pada setiap mask warna
        for category, display_name, mask, bgr_color in color_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area >= self.min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    box = (x, y, x + w, y + h)

                    # Titik tengah objek
                    cx = x + w // 2
                    cy = y + h // 2

                    detected_candidates.append({
                        "category": category,
                        "display_name": display_name,
                        "area": area,
                        "box": box,
                        "center": (cx, cy),
                        "contour": c,
                        "bgr_color": bgr_color
                    })

                    # Gambar bounding box dan label pada kontur
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), bgr_color, 2)
                    cv2.circle(annotated_frame, (cx, cy), 4, bgr_color, -1)
                    cv2.putText(
                        annotated_frame,
                        f"{display_name} ({int(area)})",
                        (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        bgr_color,
                        2
                    )

        # 5. Tentukan target terbaik (objek terbesar)
        if detected_candidates:
            detected_candidates.sort(key=lambda item: item["area"], reverse=True)
            best = detected_candidates[0]
            raw_action = best["category"]
            label_text = f"{best['display_name']} ({int(best['area'])})"
            banner_color = best["bgr_color"]
            best_box = best["box"]
        else:
            raw_action = "NONE"
            label_text = "Tidak Ada Objek"
            banner_color = (80, 80, 80)
            best_box = None

        return raw_action, label_text, annotated_frame, best_box, banner_color


# Kompatibilitas backwards dengan fungsi process_frame sebelumnya
_default_detector = None

def process_frame(frame):
    """Fungsi pembantu untuk kompatibilitas fungsi lama."""
    global _default_detector
    if _default_detector is None:
        _default_detector = ColorDetectorOpenCV()
    raw_action, label_text, annotated_frame, _, _ = _default_detector.detect(frame)
    return label_text, annotated_frame


def async_upload_photo(server_url, frame):
    """Mengunggah foto kotak biru ke server web monitoring secara background thread."""
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
    """Mengirim update status terkini ke server web monitoring."""
    def _send():
        try:
            url = f"{server_url.rstrip('/')}/api/status"
            requests.post(url, json={"object": obj_name, "led": led_state}, timeout=1.5)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Deteksi Warna OpenCV Terintegrasi - Gamantaray")
    parser.add_argument("--device", type=int, default=0, help="Index kamera (default: 0)")
    parser.add_argument("--server", type=str, default="http://localhost:5000", help="URL Web Server Monitoring")
    parser.add_argument("--no-gui", action="store_true", help="Jalankan mode headless / terminal tanpa GUI")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_CONTOUR_AREA, help="Luas kontur minimal (default: 1000)")
    args = parser.parse_args()

    has_display = ("DISPLAY" in os.environ) or (sys.platform == "win32")
    show_gui = (not args.no_gui) and has_display

    print("==================================================")
    print("DETEKSI WARNA OPENCV TERINTEGRASI - GAMANTARAY")
    print(f"Metode Deteksi   : OpenCV Color Masking (HSV)")
    print(f"Kamera Index     : {args.device}")
    print(f"Server Monitoring: {args.server}")
    print(f"Min Area Kontur  : {args.min_area}")
    print(f"Mode Tampilan    : {'GUI Aktif' if show_gui else 'Headless / Terminal'}")
    print("==================================================")

    detector = ColorDetectorOpenCV(min_area=args.min_area)
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
    last_sent_action = None

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
                active_led_state = raw_action
                last_detected_action = raw_action
                last_detect_time = now
            else:
                # Tahan nyala LED selama LED_HOLD_TIME_SEC jika 1-2 frame lolos
                if (now - last_detect_time) > LED_HOLD_TIME_SEC:
                    active_led_state = "NONE"

            # 1. Kontrol Aktuator LED GPIO
            led.set_state(active_led_state)

            # 2. Teks Status
            if raw_action != "NONE":
                status_display = label
            else:
                status_display = "Tidak Ada Objek" if active_led_state == "NONE" else f"{last_detected_action} (Holding)"

            # 3. Kirim update status ke server jika ada perubahan status
            if active_led_state != last_sent_action:
                print(f"[STATUS] {status_display:30} ---> LED: {active_led_state}")
                async_update_status(args.server, status_display, active_led_state)
                last_sent_action = active_led_state

            # 4. Logika Khusus Kotak Biru: Ambil foto dan kirim ke website
            if raw_action == "BLUE" or active_led_state == "BLUE":
                if (now - last_upload_time) > UPLOAD_COOLDOWN_SEC:
                    print("[TRIGGER] Kotak Biru terdeteksi! Mengirim foto ke website...")
                    async_upload_photo(args.server, annotated_frame)
                    last_upload_time = now

            # 5. Tampilkan banner status di bagian atas
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
                cv2.imshow("Gamantaray - Deteksi Warna OpenCV", annotated_frame)
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
