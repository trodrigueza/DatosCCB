from flask import Flask, render_template, send_from_directory, jsonify
import os

app = Flask(__name__)

VISUALIZACIONES_DIR = "visualizaciones"

@app.route("/")
def index():
    archivos = [f for f in os.listdir(VISUALIZACIONES_DIR) if f.endswith(".html")]
    return render_template("index.html", archivos=archivos)

@app.route("/visualizaciones/<path:filename>")
def visualizar_archivo(filename):
    return send_from_directory(VISUALIZACIONES_DIR, filename)

@app.route("/api/archivos")
def obtener_archivos():
    archivos = [f for f in os.listdir(VISUALIZACIONES_DIR) if f.endswith(".html")]
    return jsonify(archivos)

if __name__ == "__main__":
    app.run(debug=True)
