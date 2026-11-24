"""
Modul Deteksi Objek - Visi Komputer
Sistem Kendali & Deteksi Objek Gamantaray

Mendeteksi:
1. Bola Merah
2. Bola Hijau
3. Kotak Biru
4. Tidak Ada Objek (NONE)
"""

import cv2
import numpy as np

# ==========================================================
# 1. RENTANG WARNA HSV (Bisa disesuaikan dengan kondisi cahaya)
# ==========================================================
# Warna Merah berada di dua ujung spektrum Hue (0-10 & 170-180)
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])

# Warna Hijau
LOWER_GREEN = np.array([35, 80, 60])
UPPER_GREEN = np.array([85, 255, 255])

# Warna Biru
LOWER_BLUE = np.array([95, 120, 60])
UPPER_BLUE = np.array([135, 255, 255])

# Ambang batas luas kontur minimum (agar noise kecil tidak dianggap objek)
MIN_CONTOUR_AREA = 1000


def is_circle(contour):
    """
    Mengecek apakah kontur mendekati bentuk lingkaran (bola).
    Menggunakan rumus circularity = 4 * pi * Area / (Perimeter^2)
    Lingkaran sempurna memiliki circularity mendekati 1.0.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    # Rasio perbandingan area kontur dengan lingkaran pembungkus terkecil
    (_, _), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (radius ** 2)
    extent = area / circle_area if circle_area > 0 else 0

    return (circularity > 0.65) and (extent > 0.65)


def is_square(contour):
    """
    Mengecek apakah kontur mendekati bentuk kotak/persegi.
    1. Poligon aproksimasi memiliki 4 sudut.
    2. Rasio lebar terhadap tinggi (aspect ratio) mendekati 1:1.
    3. Kontur cembung (convex).
    """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        # Aspek rasio antara 0.75 s/d 1.33 menandakan bentuk kotak/kubus
        if 0.7 <= aspect_ratio <= 1.35:
            # Rasio luas kontur terhadap kotak pembungkus (extent)
            area = cv2.contourArea(contour)
            extent = area / float(w * h)
            if extent > 0.65:
                return True
    return False


def process_frame(frame):
    """
    Menganalisis frame kamera dan menentukan objek yang terdeteksi:
    Mengembalikan: (label_terdeteksi, visual_frame, metadata)
    """
    # 1. Bersihkan noise dengan Gaussian Blur
    blurred = cv2.GaussianBlur(frame, (9, 9), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. Buat Masking untuk masing-masing warna
    # Mask Merah (kombinasi dua range Hue)
    mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Mask Hijau
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # Mask Biru
    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # Operasi morfologi (hilangkan bintik-bintik kecil)
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

    detected_label = "TIDAK ADA OBJEK"
    detected_color_bgr = (200, 200, 200)

    candidates = []

    # Periksa Kontur Merah -> Cek apakah Bola
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_red:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA and is_circle(c):
            candidates.append(("Bola Merah", area, c, (0, 0, 255)))

    # Periksa Kontur Hijau -> Cek apakah Bola
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_green:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA and is_circle(c):
            candidates.append(("Bola Hijau", area, c, (0, 255, 0)))

    # Periksa Kontur Biru -> Cek apakah Kotak
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours_blue:
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA and is_square(c):
            candidates.append(("Kotak Biru", area, c, (255, 120, 0)))

    # Jika ada objek yang lolos filter, pilih objek terbesar
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        label, area, contour, bgr = candidates[0]
        detected_label = label
        detected_color_bgr = bgr

        # Gambar bounding box dan label pada kontur
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), detected_color_bgr, 3)
        cv2.putText(
            frame,
            f"{detected_label} ({int(area)})",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            detected_color_bgr,
            2,
        )

    # Gambar banner status di bagian atas frame
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (30, 30, 30), -1)
    status_text = f"Status: {detected_label}"
    cv2.putText(
        frame,
        status_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        detected_color_bgr,
        2,
    )

    return detected_label, frame


def main():
    print("==================================================")
    print("Membuka Kamera... Tekan 'q' pada jendela untuk keluar.")
    print("==================================================")

    # 0 biasanya adalah webcam default laptop/USB
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Tidak dapat mengakses kamera! Pastikan webcam tidak sedang dipakai aplikasi lain.")
        return

    # Set resolusi standar (640x480 agar ringan di Raspberry Pi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame kamera.")
            break

        # Flip horizontal agar seperti cermin (opsional tapi nyaman untuk tes webcam)
        frame = cv2.flip(frame, 1)

        label, annotated_frame = process_frame(frame)

        # Tampilkan hasil
        cv2.imshow("Gamantaray - Deteksi Objek", annotated_frame)

        # Tombol 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
