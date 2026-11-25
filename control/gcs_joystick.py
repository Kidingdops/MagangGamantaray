"""
Program Pengirim Kendali Stik PS (GCS Laptop -> Raspberry Pi)
Sistem Kendali Thruster Gamantaray

Dijalankan di Laptop (GCS) tempat Stik PS tercolok.
Membaca input analog stick lalu mengirim paket UDP ke Raspberry Pi.
"""

import sys
import time
import json
import socket
import argparse
import pygame

# Standar sinyal PWM ESC Thruster:
# 1500 us = Netral / Diam
# 1100 us = Mundur Penuh
# 1900 us = Maju Penuh
PWM_NEUTRAL = 1500
PWM_MIN = 1100
PWM_MAX = 1900
DEADZONE = 0.08  # Toleransi agar stik netral tidak bergetar


def map_axis_to_pwm(axis_val):
    """
    Mengonversi nilai axis analog (-1.0 s/d +1.0) ke rentang PWM ESC (1100 s/d 1900).
    Catatan: Pada stik PS/Xbox, dorong stik ke DEPAN bernilai negatif (-1.0).
    Jadi kita kalikan -1 agar dorong ke depan = Maju (PWM naik).
    """
    if abs(axis_val) < DEADZONE:
        return PWM_NEUTRAL

    # Invert axis: dorong ke atas (-1.0) menjadi +1.0 (maju)
    val = -axis_val

    if val > 0:
        pwm = PWM_NEUTRAL + int(val * (PWM_MAX - PWM_NEUTRAL))
    else:
        pwm = PWM_NEUTRAL + int(val * (PWM_NEUTRAL - PWM_MIN))

    return max(PWM_MIN, min(PWM_MAX, pwm))


def main():
    parser = argparse.ArgumentParser(description="GCS Joystick Controller - Gamantaray")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP Address Raspberry Pi (contoh: 192.168.1.50)")
    parser.add_argument("--port", type=int, default=5005, help="Port UDP (default: 5005)")
    parser.add_argument("--axis", type=int, default=1, help="Index Axis Maju/Mundur (default: 1 untuk Left Stick Y)")
    args = parser.parse_args()

    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("[ERROR] Tidak ada Stik PS / Gamepad yang terdeteksi di laptop!")
        print("Sambungkan stik PS melalui USB atau Bluetooth, lalu jalankan ulang.")
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print("==================================================")
    print(f"Stik Terdeteksi: {js.get_name()}")
    print(f"Target Kirim   : {args.ip}:{args.port}")
    print("Tekan Ctrl+C untuk berhenti.")
    print("==================================================")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    armed = False
    last_print = 0

    try:
        while True:
            pygame.event.pump()

            # Baca analog stick Y (Maju/Mundur)
            axis_val = js.get_axis(args.axis)
            pwm_val = map_axis_to_pwm(axis_val)

            # Tombol X (tombol 0) untuk ARM, Tombol Lingkaran (tombol 1) untuk DISARM
            # (Mapping tombol bisa berbeda antar stik, ini standar PS4/PS5)
            if js.get_button(0):  # Tombol X / Cross
                armed = True
            elif js.get_button(1):  # Tombol O / Circle
                armed = False

            # Susun paket data
            packet = {
                "pwm": pwm_val,
                "armed": armed,
                "timestamp": time.time()
            }

            msg = json.dumps(packet).encode("utf-8")
            sock.sendto(msg, (args.ip, args.port))

            # Tampilkan status di konsol setiap 0.2 detik
            now = time.time()
            if (now - last_print) > 0.2:
                arah = "DIAM (NETRAL)" if pwm_val == 1500 else ("MAJU" if pwm_val > 1500 else "MUNDUR")
                status_arm = "ARMED (AKTIF)" if armed else "DISARMED (KUNCI)"
                print(f"[GCS] Stik: {axis_val:+.2f} | PWM: {pwm_val:4d} us ({arah:13s}) | Status: {status_arm}")
                last_print = now

            time.sleep(0.05)  # 20 Hz transmission rate

    except KeyboardInterrupt:
        print("\nMenghentikan pengiriman stik... Mengirim sinyal aman NETRAL.")
        stop_packet = json.dumps({"pwm": PWM_NEUTRAL, "armed": False, "timestamp": time.time()}).encode("utf-8")
        sock.sendto(stop_packet, (args.ip, args.port))
    finally:
        sock.close()
        pygame.quit()


if __name__ == "__main__":
    main()
