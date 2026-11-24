"""
Tool Kalibrasi HSV Interaktif
Gunakan alat ini jika warna objek di ruanganmu sulit terdeteksi karena pencahayaan.
Geser slider untuk mendapatkan nilai H, S, V yang pas, lalu salin angkanya ke detect_objects.py.
"""

import cv2
import numpy as np


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Tuning HSV", cv2.WINDOW_NORMAL)

    # Buat slider (trackbar) untuk H, S, V minimum dan maksimum
    cv2.createTrackbar("H Min", "Tuning HSV", 0, 180, nothing)
    cv2.createTrackbar("H Max", "Tuning HSV", 180, 180, nothing)
    cv2.createTrackbar("S Min", "Tuning HSV", 50, 255, nothing)
    cv2.createTrackbar("S Max", "Tuning HSV", 255, 255, nothing)
    cv2.createTrackbar("V Min", "Tuning HSV", 50, 255, nothing)
    cv2.createTrackbar("V Max", "Tuning HSV", 255, 255, nothing)

    print("Gerakkan trackbar untuk menyesuaikan rentang warna.")
    print("Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ambil posisi trackbar saat ini
        h_min = cv2.getTrackbarPos("H Min", "Tuning HSV")
        h_max = cv2.getTrackbarPos("H Max", "Tuning HSV")
        s_min = cv2.getTrackbarPos("S Min", "Tuning HSV")
        s_max = cv2.getTrackbarPos("S Max", "Tuning HSV")
        v_min = cv2.getTrackbarPos("V Min", "Tuning HSV")
        v_max = cv2.getTrackbarPos("V Max", "Tuning HSV")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Masking dan hasil filter
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Gabungkan tampilan frame asli dan hasil mask berdampingan
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((frame, mask_bgr, result))
        resized = cv2.resize(stacked, (960, 240))

        cv2.imshow("Tuning HSV", resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n--- Nilai Terakhir yang Dipilih ---")
            print(f"LOWER = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"UPPER = np.array([{h_max}, {s_max}, {v_max}])")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
