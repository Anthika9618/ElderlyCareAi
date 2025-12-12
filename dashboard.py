from flask import Flask, render_template, jsonify, request, Response
import pymysql
import os
from functools import wraps

app = Flask(__name__, template_folder="templates")

# ======= Basic Auth Config =======
AUTH_USERNAME = os.getenv("DASHBOARD_USER", "anthikaForthtrack")
AUTH_PASSWORD = os.getenv("DASHBOARD_PASS", "aunForthtrack0263")

def check_auth(username, password):
    return username == AUTH_USERNAME and password == AUTH_PASSWORD

def authenticate():
    return Response(
        'Unauthorized access. Please provide valid credentials.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
# ==================================

# DB config
DB_HOST = os.getenv("DB_HOST", "192.168.1.62")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "W3r4dm1n")
DB_NAME = os.getenv("DB_NAME", "mqtt_device")

def get_db_connection():
    conn = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn

# หน้า Dashboard ต้อง login ก่อน
@app.route("/")
@requires_auth
def dashboard():
    return render_template("dashboard.html")

# API device logs
@app.route("/api/device_logs")
@requires_auth
def api_device_logs():
    logs = []
    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 50))
        keyword = request.args.get("keyword", "").strip()

        where_clauses = []
        params = []

        if keyword:
            like = f"%{keyword}%"
            where_clauses.append("(" +
                " OR ".join([
                    "device_id LIKE %s",
                    "mac LIKE %s",
                    "state LIKE %s",
                    "temp LIKE %s",
                    "hum LIKE %s",
                    "rssi LIKE %s",
                    "battery LIKE %s",
                    "hart_rate LIKE %s",
                    "breath_rate LIKE %s"
                ]) +
            ")")
            params += [like] * 9

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        offset = (page - 1) * per_page

        sql = f"""
            SELECT device_id, mac, state, temp, hum, rssi, battery, movecount, pitch, roll, human, fall, hart_rate, breath_rate, distance, time_stamp
            FROM device_info
            {where_sql}
            ORDER BY time_stamp DESC
            LIMIT %s OFFSET %s
        """
        params += [per_page, offset]

        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            logs = cursor.fetchall()

    except Exception as e:
        print("❌ Error fetching data:", e)
        logs = []
    finally:
        try:
            conn.close()
        except:
            pass

    return jsonify(logs)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
