"""
Program Jembatan Kendali Thruster (Raspberry Pi -> Pixhawk)
Sistem Kendali Thruster Gamantaray

Dijalankan di Raspberry Pi.
Menerima paket UDP dari GCS Laptop, lalu meneruskannya ke Pixhawk via MAVLink.
Dilengkapi Failsafe otomatis:
- Jika koneksi stik putus (> 0.5s tanpa data) -> Thruster OTOMATIS DIAM (PWM 1500 us).
- Jika stik di posisi netral -> Thruster DIAM.
"""

import sys
import time
import json
import socket
import argparse
from pymavlink import mavutil

PWM_NEUTRAL = 1500
FAILSAFE_TIMEOUT_SEC = 0.5  # Batas waktu paket hilang sebelum failsafe aktif


class PixhawkController:
    def __init__(self, connection_string, baudrate=115200, mock=False):
        self.mock = mock
        self.master = None
        self.is_armed = False

        if not self.mock:
            print(f"[MAVLINK] Menghubungkan ke Pixhawk di {connection_string} (Baud: {baudrate})...")
            try:
                self.master = mavutil.mavlink_connection(connection_string, baud=baudrate)
                print("[MAVLINK] Menunggu Heartbeat dari Pixhawk...")
                self.master.wait_heartbeat(timeout=5)
                print(f"[MAVLINK] Terhubung ke Pixhawk! (Sistem ID: {self.master.target_system})")
            except Exception as e:
                print(f"[WARN] Gagal terhubung ke Pixhawk fisik: {e}")
                print("[INFO] Beralih ke MODE SIMULASI (MOCK) agar pengujian tetap bisa berjalan.")
                self.mock = True
        else:
            print("[INFO] Berjalan dalam MODE SIMULASI (MOCK) tanpa Pixhawk fisik.")

    def set_arm(self, arm_state):
        if self.mock or not self.master:
            self.is_armed = arm_state
            return

        if arm_state != self.is_armed:
            param = 1 if arm_state else 0
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                param, 0, 0, 0, 0, 0, 0
            )
            self.is_armed = arm_state
            print(f"[MAVLINK] Perintah {'ARM' if arm_state else 'DISARM'} dikirim ke Pixhawk.")

    def send_thruster_pwm(self, pwm_val, channel=3):
        """
        Mengirim sinyal PWM ke ESC melalui channel RC Override.
        channel: nomor output motor di Pixhawk (default Channel 3 atau Channel 1).
        """
        if self.mock or not self.master:
            return

        # 65535 berarti abaikan channel tersebut (tidak di-override)
        rc_channel_values = [65535] * 8
        rc_channel_values[channel - 1] = int(pwm_val)

        self.master.mav.rc_channels_override_send(
            self.master.target_system,
            self.master.target_component,
            *rc_channel_values
        )


def main():
    parser = argparse.ArgumentParser(description="Pixhawk Thruster Bridge - Gamantaray")
    parser.add_argument("--port", type=int, default=5005, help="Port UDP penerima (default: 5005)")
    parser.add_argument("--device", type=str, default="/dev/ttyACM0", help="Port serial Pixhawk (contoh: /dev/ttyACM0 atau /dev/serial0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baudrate serial (default: 115200)")
    parser.add_argument("--channel", type=int, default=3, help="Channel output motor ESC (default: 3)")
    parser.add_argument("--mock", action="store_true", help="Paksa mode simulasi tanpa Pixhawk")
    args = parser.parse_args()

    pix = PixhawkController(args.device, baudrate=args.baud, mock=args.mock)

    # Siapkan UDP Socket Receiver
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)

    print("==================================================")
    print("PIXHAWK THRUSTER BRIDGE BERJALAN")
    print(f"Mendengarkan UDP di port: {args.port}")
    print(f"Channel Output ESC     : Channel {args.channel}")
    print("Proteksi Failsafe Aktif : Timeout 0.5s")
    print("Tekan Ctrl+C untuk keluar.")
    print("==================================================")

    last_packet_time = time.time()
    current_pwm = PWM_NEUTRAL
    is_failsafe_active = False

    try:
        while True:
            # 1. Baca paket UDP yang masuk
            try:
                data, addr = sock.recvfrom(1024)
                packet = json.loads(data.decode("utf-8"))

                current_pwm = packet.get("pwm", PWM_NEUTRAL)
                arm_request = packet.get("armed", False)
                last_packet_time = time.time()

                if is_failsafe_active:
                    print(f"[RECONNECTED] Sinyal dari {addr[0]} kembali terhubung!")
                    is_failsafe_active = False

                pix.set_arm(arm_request)

            except BlockingIOError:
                pass  # Tidak ada paket baru di siklus ini
            except Exception as e:
                pass

            # 2. Periksa Kondisi Failsafe (Koneksi Terputus)
            now = time.time()
            if (now - last_packet_time) > FAILSAFE_TIMEOUT_SEC:
                if not is_failsafe_active:
                    print("[FAILSAFE AKTIF] Sinyal stik terputus! Mengunci thruster ke NETRAL (1500 us).")
                    is_failsafe_active = True
                current_pwm = PWM_NEUTRAL

            # 3. Kirim sinyal PWM ke Pixhawk
            pix.send_thruster_pwm(current_pwm, channel=args.channel)

            time.sleep(0.02)  # Loop rate 50 Hz

    except KeyboardInterrupt:
        print("\nMematikan jembatan kendali... Menghentikan thruster.")
    finally:
        pix.send_thruster_pwm(PWM_NEUTRAL, channel=args.channel)
        pix.set_arm(False)
        sock.close()


if __name__ == "__main__":
    main()
