import os
import time
import json
import pymysql
import warnings
import paho.mqtt.client as mqtt

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================================
# CONFIG
# ====================================
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.1.66")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", "aunmqtt")
MQTT_PASS = os.getenv("MQTT_PASS", "mqtt7890")

DB_HOST = os.getenv("DB_HOST", "192.168.1.66")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "W3r4dm1n")
DB_NAME = os.getenv("DB_NAME", "mqtt_device")

TOPIC_DEVICE = "forth/customer/device"
TOPIC_CONFIG = "device/{}/config"
TOPIC_REPLY = "device/{}/reply"
TOPIC_BLE_CONFIG = "config/ble_devices"

# ====================================
# DATABASE CONNECTION
# ====================================
def get_db_connection():
    for i in range(10):
        try:
            conn = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor
            )
            return conn
        except Exception as e:
            print(f"⚠️ DB not ready ({e}), retrying in 3s...")
            time.sleep(3)
    raise Exception("❌ Cannot connect to MySQL after 10 retries")

# ====================================
# SAVE DEVICE DATA
# ====================================

def save_device_data(db, data, raw_message):
    device_id = data.get("deviceId", "Unknown")
    time_stamp = data.get("Time", None)
    sensors = data.get("Sensor", [])

    with db.cursor() as cursor:
        for s in sensors:
            # ดึงค่า Battery (mV)
            battery_mv = int(s.get("Battery", 0))
            
            # แปลงค่าเป็นเปอร์เซ็นต์ (สมมติ 3000mV = 0%, 4200mV = 100%)
            battery_percent = (battery_mv - 3000) / 1200 * 100
            battery_percent = max(0, min(100, battery_percent))  # จำกัดให้อยู่ 0–100%
            battery_percent = round(battery_percent)  # ปัดเป็นจำนวนเต็ม

            cursor.execute("""
                INSERT INTO device_info 
                (device_id, mac, state, type, temp, hum, rssi, battery, battery_percent, movecount, pitch, roll, human, fall, hart_rate, breath_rate, distance, time_stamp, raw_message)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                device_id,
                s.get("Mac"),
                s.get("State"),
                s.get("Type"),
                float(s.get("Temp", 0)),
                float(s.get("Hum", 0)),
                int(s.get("RSSI", 0)),
                battery_mv,
                battery_percent,
                int(s.get("MoveCount", 0)),
                int(s.get("Pitch", 0)),
                int(s.get("Roll", 0)),
                int(s.get("Human", 0)),
                int(s.get("Fall", 0)),
                int(s.get("HartRate", 0)),
                int(s.get("BreathRate", 0)),
                float(s.get("Distance", 0)),
                time_stamp,
                raw_message
            ))
            print(f"✅ Device {device_id} Sensor {s.get('Mac')} => Battery: {battery_mv} mV = {battery_percent}%")

    db.commit()

# ====================================
# MQTT CALLBACKS
# ====================================
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("✅ Connected to MQTT broker!")
        client.subscribe(TOPIC_DEVICE)
        client.subscribe("device/+/reply")
        client.subscribe("device/+/ack")
        print("📡 Subscribed to device topics.")
    else:
        print(f"❌ MQTT connect failed: {reason_code}")

ack_flags = {}

def on_message(client, userdata, msg):
    global ack_flags
    topic = msg.topic
    payload = msg.payload.decode(errors="ignore")
    print(f"📩 Message from {topic}: {payload}")

    # รับ ack OK
    if topic.startswith("device/") and topic.endswith("/reply") and "OK" in payload.upper():
        device_id = topic.split("/")[1]
        print(f"✅ Device {device_id} acknowledged config.")
        db = get_db_connection()
        with db.cursor() as cursor:
            cursor.execute(
                "UPDATE wait_command SET status='done', updated_at=NOW() WHERE device_id=%s AND status='pending'",
                (device_id,)
            )
            db.commit()
        db.close()
        ack_flags[device_id] = True
        return

    # รับข้อมูล Device
    if topic.startswith("forth/customer/device"):
        try:
            data = json.loads(payload)
            db = get_db_connection()
            save_device_data(db, data, payload)
            db.close()
        except Exception as e:
            print(f"❌ Error processing message: {e}")

# ====================================
# SEND PENDING COMMANDS
# ====================================
def send_pending_commands(client):
    global ack_flags
    try:
        db = get_db_connection()
        with db.cursor() as cursor:
            cursor.execute("SELECT * FROM wait_command WHERE status='pending'")
            commands = cursor.fetchall()
            for cmd in commands:
                device_id = cmd["device_id"]
                config = json.loads(cmd["command"])
                topic = TOPIC_CONFIG.format(device_id)
                ack_flags[device_id] = False
                print(f"📤 Sending config to {device_id}: {config}")

                for i in range(10):
                    if ack_flags.get(device_id):
                        print(f"✅ Device {device_id} already replied OK!, stop sending.")
                        break
                    client.publish(topic, json.dumps(config))
                    print(f"🔁 Round {i+1}/10: publishing to {topic}")
                    time.sleep(15)

                    if ack_flags.get(device_id):
                        break
                else:
                    print(f"❌ Device {device_id} no response after 10 rounds.")
                    cursor.execute(
                        "UPDATE wait_command SET status='failed', updated_at=NOW() WHERE id=%s",
                        (cmd["id"],)
                    )
                    db.commit()

                ack_flags.pop(device_id, None)

    except Exception as e:
        print(f"❌ Error sending config: {e}")
    finally:
        db.close()

# ====================================
# SEND BLE CONFIG FROM DATABASE
# ====================================
def send_ble_config_once(client):
    try:
        db = get_db_connection()
        with db.cursor() as cursor:
            cursor.execute("SELECT detail FROM device_config")
            rows = cursor.fetchall()

        room_config = [json.loads(row['detail']) for row in rows]

        payload = json.dumps(room_config)
        client.publish(TOPIC_BLE_CONFIG, payload)
        print(f"📤 ส่งข้อมูล BLE Config ไปยัง {TOPIC_BLE_CONFIG}: {payload}")

    except Exception as e:
        print(f"❌ Error sending BLE Config: {e}")
    finally:
        db.close()

# ====================================
# MAIN
# ====================================
def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(MQTT_USER, MQTT_PASS)

    for attempt in range(15):
        try:
            print(f"🔄 Connecting MQTT... (try {attempt+1})")
            client.connect(MQTT_BROKER, MQTT_PORT, 60)
            break
        except Exception as e:
            print(f"⚠️ {e} — retrying in 3s...")
            time.sleep(3)
    else:
        print("❌ Could not connect to MQTT broker after retries.")
        return

    print("🚀 MQTT app running...")
    client.loop_start()
    time.sleep(2)
    send_ble_config_once(client)

    while True:
        send_pending_commands(client)
        time.sleep(10)

if __name__ == "__main__":
    main()
