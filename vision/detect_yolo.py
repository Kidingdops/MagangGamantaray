"""
Sistem Deteksi Objek Terintegrasi - Gamantaray (v3 - OpenCV Color Detection)
Menghubungkan:
1. Kamera + OpenCV Color Detection (Bola Merah, Bola Hijau, Kotak Biru)
2. Aktuator LED GPIO (Merah, Hijau, Biru) - dengan State Persistence (Anti-Kedip)
3. Website Monitoring (Upload Otomatis Saat Kotak Biru Terdeteksi)

Catatan:
Deteksi telah dimigrasikan dari model YOLO (.pt) ke OpenCV HSV Color Detection
agar lebih cepat, ringan, dan stabil di Raspberry Pi.
"""

import os
import sys
import time
import argparse
import threading
import cv2
import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hardware.led_controller import LEDController
from vision.detect_objects import (
    ColorDetectorOpenCV,
    async_upload_photo,
    async_update_status,
    DEFAULT_MIN_CONTOUR_AREA,
    UPLOAD_COOLDOWN_SEC,
    LED_HOLD_TIME_SEC
)

# Alias ObjectDetectorAI ke ColorDetectorOpenCV untuk kompatibilitas kode
ObjectDetectorAI = ColorDetectorOpenCV


def main():
    parser = argparse.ArgumentParser(description="Deteksi Objek Terintegrasi Gamantaray (OpenCV)")
    parser.add_argument("--device", type=int, default=0, help="Index kamera (default: 0)")
    parser.add_argument("--server", type=str, default="http://localhost:5000", help="URL Web Server Monitoring")
    parser.add_argument("--no-gui", action="store_true", help="Jalankan mode headless / SSH")
    parser.add_argument("--min-area", type=int, default=DEFAULT_MIN_CONTOUR_AREA, help="Luas kontur minimal (default: 1000)")
    # Tetap sediakan flag --conf untuk backwards compatibility argumen CLI
    parser.add_argument("--conf", type=float, default=0.35, help="Ambang batas (diabaikan pada mode OpenCV)")
    args = parser.parse_args()

    has_display = ("DISPLAY" in os.environ) or (sys.platform == "win32")
    show_gui = (not args.no_gui) and has_display

    print("==================================================")
    print("SISTEM INTEGRASI KENDALI & DETEKSI OBJEK GAMANTARAY")
    print("Metode           : OpenCV Color Detection (HSV)")
    print(f"Kamera Index     : {args.device}")
    print(f"Server Monitoring: {args.server}")
    print(f"Min Area Kontur  : {args.min_area}")
    print(f"Mode Display     : {'GUI Aktif' if show_gui else 'Headless / Terminal'}")
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
                if (now - last_detect_time) > LED_HOLD_TIME_SEC:
                    active_led_state = "NONE"

            # 1. Update ke Hardware LED GPIO
            led.set_state(active_led_state)

            # 2. Update status ke terminal & website jika ada perubahan
            if raw_action != "NONE":
                status_display = label
            else:
                status_display = "Tidak Ada Objek" if active_led_state == "NONE" else f"{last_detected_action} (Holding)"

            if active_led_state != last_sent_action:
                print(f"[STATUS] {status_display:30} ---> LED: {active_led_state}")
                async_update_status(args.server, status_display, active_led_state)
                last_sent_action = active_led_state

            # 3. Logika Khusus Kotak Biru: Ambil foto dan kirim ke website
            if raw_action == "BLUE" or active_led_state == "BLUE":
                if (now - last_upload_time) > UPLOAD_COOLDOWN_SEC:
                    print("[TRIGGER] Kotak Biru terdeteksi stabil! Mengirim foto ke website...")
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
