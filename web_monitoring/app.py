"""
Server Website Monitoring Gamantaray
Menampilkan status deteksi dan galeri foto objek Kotak Biru secara realtime.
"""

import os
import glob
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# Direktori penyimpanan foto yang diunggah
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Status global sistem
current_system_state = {
    "detected_object": "Belum ada data",
    "active_led": "NONE",
    "last_update": datetime.now().strftime("%H:%M:%S"),
    "total_photos": 0
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Endpoint menerima foto saat kamera mendeteksi Kotak Biru.
    """
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "File tidak ditemukan"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Nama file kosong"}), 400

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"kotak_biru_{timestamp_str}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    current_system_state["detected_object"] = "Kotak Biru"
    current_system_state["active_led"] = "BLUE"
    current_system_state["last_update"] = datetime.now().strftime("%H:%M:%S")

    print(f"[WEB SERVER] Foto baru diterima & disimpan: {filename}")
    return jsonify({"status": "success", "filename": filename}), 200


@app.route("/api/status", methods=["GET", "POST"])
def api_status():
    """
    Update atau ambil status deteksi saat ini.
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        if "object" in data:
            current_system_state["detected_object"] = data["object"]
        if "led" in data:
            current_system_state["active_led"] = data["led"]
        current_system_state["last_update"] = datetime.now().strftime("%H:%M:%S")
        return jsonify({"status": "updated"})

    # GET: Kembalikan status dan daftar foto terbaru
    image_files = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"], "*.jpg"))
    # Urutkan berdasarkan waktu modifikasi terbaru
    image_files.sort(key=os.path.getmtime, reverse=True)

    photos = []
    for img_path in image_files[:12]:  # Ambil maksimal 12 foto terbaru
        fname = os.path.basename(img_path)
        mtime = datetime.fromtimestamp(os.path.getmtime(img_path)).strftime("%Y-%m-%d %H:%M:%S")
        photos.append({"filename": fname, "timestamp": mtime, "url": f"/uploads/{fname}"})

    current_system_state["total_photos"] = len(image_files)

    return jsonify({
        "status": current_system_state,
        "photos": photos
    })


if __name__ == "__main__":
    # Host 0.0.0.0 agar bisa diakses dari laptop dan Raspberry Pi dalam satu jaringan WiFi
    print("==================================================")
    print("Server Website Monitoring Gamantaray Berjalan!")
    print("Akses melalui browser di: http://localhost:5000")
    print("==================================================")
    app.run(host="0.0.0.0", port=5000, debug=False)
