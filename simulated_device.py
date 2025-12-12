import json
import time
import random
import paho.mqtt.client as mqtt

# ===== CONFIG =====
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883

TOPIC_DEVICE = "forth/customer/device"
TOPIC_CONFIG_PREFIX = "device/{}/config"
TOPIC_REPLY_PREFIX = "device/{}/reply"

# ===== DEVICE INFO =====
device_name = random.choice(["Eyesensor", "Fallsensor"])
device_id = f"{device_name}-{random.randint(1, 100)}"  
imei = str(random.randint(1000000000, 9999999999))

# ===== MQTT SETUP =====
client = mqtt.Client()

def send_device_info():
    status = random.choice(["active", "idle", "error"])
    payload = {
        "device_id": device_id,
        "customer_id": random.randint(1, 5),
        "device_name": device_name,
        "imei": imei,
        "status": status
    }
    client.publish(TOPIC_DEVICE, json.dumps(payload))
    print(f"📤 Sent device info: {payload}")

def on_message(client, userdata, msg):
    topic = msg.topic
    payload = msg.payload.decode()
    print(f"📩 Received message on {topic}: {payload}")

    # ถ้าเป็น config → ตอบกลับ OK!
    if topic == TOPIC_CONFIG_PREFIX.format(device_id):
        time.sleep(random.randint(1, 3))
        reply_topic = TOPIC_REPLY_PREFIX.format(device_id)
        client.publish(reply_topic, "OK!")
        print(f"📤 Sent ACK to {reply_topic}")

# subscribe topic config
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
config_topic = TOPIC_CONFIG_PREFIX.format(device_id)
client.subscribe(config_topic)

client.loop_start()
print(f"🚀 Simulated device '{device_name}' with ID '{device_id}' running...")

while True:
    send_device_info()
    time.sleep(random.randint(5, 10))
