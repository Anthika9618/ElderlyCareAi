import paho.mqtt.client as mqtt
import psycopg2
import json
import datetime


conn = psycopg2.connect(
    host="localhost",
    dbname="sensor_db",
    user="aunaun",  
    password="08547086340263"
)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS sensor_data (
        id SERIAL PRIMARY KEY,
        device_id TEXT,
        device_name TEXT,
        imei TEXT,
        status TEXT,
        datetime TEXT
    )
''')
conn.commit()


def on_connect(client, userdata, flags, rc):
    print("✅ MQTT Connected with result code", rc)
    client.subscribe("sensor/fall")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"\n📩 Received from {payload['device_name']}:")
        print(f"  Device ID: {payload['device_id']}")
        print(f"  IMEI: {payload['imei']}")
        print(f"  Status: {payload['status']}")
        print(f"  DateTime: {payload['datetime']}")


        cursor.execute('''
            INSERT INTO sensor_data (device_id, device_name, imei, status, datetime)
            VALUES (%s, %s, %s, %s, %s)
        ''', (payload['device_id'], payload['device_name'], payload['imei'], payload['status'], payload['datetime']))
        conn.commit()

    except Exception as e:
        conn.rollback()  # ป้องกัน transaction ถูกล็อค
        print("❌ Insert Error:", e)

# ----------------------------
# MQTT Subscriber
# ----------------------------
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

BROKER_IP = "broker.hivemq.com"
client.connect(BROKER_IP, 1883, 60)

print("🚀 Waiting for sensor data...")
client.loop_forever()
