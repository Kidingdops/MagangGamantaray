"""
Modul Kontrol Aktuator LED GPIO Raspberry Pi 4B
Sistem Kendali & Deteksi Objek Gamantaray

Pemetaan Pin BCM:
- LED Merah : GPIO 17 (Pin Fisik 11)
- LED Hijau : GPIO 27 (Pin Fisik 13)
- LED Biru  : GPIO 22 (Pin Fisik 15)
- Ground    : GND (Pin Fisik 6 / 9 / 14 / 20)
"""

import time
import sys

# Definisi Pin BCM (Broadcom GPIO Numbering)
PIN_LED_RED = 17    # Pin Fisik 11
PIN_LED_GREEN = 27  # Pin Fisik 13
PIN_LED_BLUE = 22   # Pin Fisik 15

# Flag apakah berjalan di hardware asli Raspberry Pi
IS_RASPBERRY_PI = False
gpio_module = None

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PIN_LED_RED, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_LED_GREEN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PIN_LED_BLUE, GPIO.OUT, initial=GPIO.LOW)
    IS_RASPBERRY_PI = True
    gpio_module = GPIO
    print("[HARDWARE] Terhubung ke GPIO Raspberry Pi asli.")
except (ImportError, RuntimeError):
    print("[SIMULASI] Library RPi.GPIO tidak tersedia. Berjalan dalam mode simulasi.")
    IS_RASPBERRY_PI = False


class LEDController:
    def __init__(self):
        self.current_state = "NONE"
        self.turn_off_all()

    def turn_off_all(self):
        """Mematikan semua LED (Merah, Hijau, Biru)"""
        if IS_RASPBERRY_PI and gpio_module:
            gpio_module.output(PIN_LED_RED, gpio_module.LOW)
            gpio_module.output(PIN_LED_GREEN, gpio_module.LOW)
            gpio_module.output(PIN_LED_BLUE, gpio_module.LOW)
        self.current_state = "NONE"

    def set_red(self):
        """Menyalakan LED Merah, mematikan yang lain"""
        if IS_RASPBERRY_PI and gpio_module:
            gpio_module.output(PIN_LED_RED, gpio_module.HIGH)
            gpio_module.output(PIN_LED_GREEN, gpio_module.LOW)
            gpio_module.output(PIN_LED_BLUE, gpio_module.LOW)
        self.current_state = "RED"

    def set_green(self):
        """Menyalakan LED Hijau, mematikan yang lain"""
        if IS_RASPBERRY_PI and gpio_module:
            gpio_module.output(PIN_LED_RED, gpio_module.LOW)
            gpio_module.output(PIN_LED_GREEN, gpio_module.HIGH)
            gpio_module.output(PIN_LED_BLUE, gpio_module.LOW)
        self.current_state = "GREEN"

    def set_blue(self):
        """Menyalakan LED Biru, mematikan yang lain"""
        if IS_RASPBERRY_PI and gpio_module:
            gpio_module.output(PIN_LED_RED, gpio_module.LOW)
            gpio_module.output(PIN_LED_GREEN, gpio_module.LOW)
            gpio_module.output(PIN_LED_BLUE, gpio_module.HIGH)
        self.current_state = "BLUE"

    def set_state(self, state):
        """
        Menerima state deteksi: 'RED', 'GREEN', 'BLUE', atau 'NONE'
        Sesuai tabel tugas:
        - Bola Merah  -> LED Merah
        - Bola Hijau  -> LED Hijau
        - Kotak Biru  -> LED Biru
        - Tidak Ada   -> Semua LED Mati
        """
        state_upper = state.upper() if state else "NONE"
        if state_upper == self.current_state:
            return  # Tidak perlu ubah state jika sama

        if state_upper == "RED":
            self.set_red()
        elif state_upper == "GREEN":
            self.set_green()
        elif state_upper == "BLUE":
            self.set_blue()
        else:
            self.turn_off_all()

    def cleanup(self):
        """Membersihkan konfigurasi pin saat keluar"""
        self.turn_off_all()
        if IS_RASPBERRY_PI and gpio_module:
            gpio_module.cleanup()
        print("[HARDWARE] GPIO Cleanup selesai.")


def test_led():
    """Fungsi pengujian mandiri untuk memastikan LED menyala bergantian"""
    print("\n================================================")
    print("Memulai Pengujian Urutan LED (Self-Test)")
    print("Urutan: MERAH -> HIJAU -> BIRU -> MATI SEMUA")
    print("Tekan Ctrl+C untuk berhenti.")
    print("================================================\n")

    led = LEDController()
    try:
        while True:
            print(">> [TEST] Nyalakan LED MERAH...")
            led.set_red()
            time.sleep(1.5)

            print(">> [TEST] Nyalakan LED HIJAU...")
            led.set_green()
            time.sleep(1.5)

            print(">> [TEST] Nyalakan LED BIRU...")
            led.set_blue()
            time.sleep(1.5)

            print(">> [TEST] Matikan SEMUA LED...")
            led.turn_off_all()
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("\nPengujian dihentikan.")
    finally:
        led.cleanup()


if __name__ == "__main__":
    test_led()
