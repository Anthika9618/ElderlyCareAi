from flask import Flask, render_template, jsonify
import psycopg2

app = Flask(__name__)

# ----------------------------
# Database
# ----------------------------
conn = psycopg2.connect(
    host="localhost",
    dbname="sensor_db",
    user="aunaun",
    password="08547086340263"
)
cursor = conn.cursor()

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("app_mq.html")

@app.route("/api/data")
def get_data():
    cursor.execute("SELECT device_id, device_name, imei, status, datetime FROM sensor_data ORDER BY id DESC LIMIT 100")
    rows = cursor.fetchall()

    sensors = {}
    for device_id, device_name, imei, status, dt in reversed(rows):
        if device_name not in sensors:
            sensors[device_name] = {"device_id": [], "imei": [], "status": [], "datetime": []}
        sensors[device_name]["device_id"].append(device_id)
        sensors[device_name]["imei"].append(imei)
        sensors[device_name]["status"].append(status)
        sensors[device_name]["datetime"].append(dt)

    return jsonify(sensors)

# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
