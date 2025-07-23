import paho.mqtt.client as mqtt
import time
import random
import json

broker_ip = "192.168.79.146"
broker_port = 1883
topic = "sensor/fall_detection"

client = mqtt.Client()
client.connect(broker_ip, broker_port, 60)
client.loop_start()

try:
    while True:
        message = {
            "fall_detected": random.choice([True, False]),
            "timestamp": int(time.time())
        }
        msg_str = json.dumps(message)
        result = client.publish(topic, msg_str)
        status = result[0]
        if status == 0:
            print(f"Sent message to topic {topic}: {msg_str}")
        else:
            print(f"Failed to send message to topic {topic}")
        time.sleep(5)
except KeyboardInterrupt:
    client.disconnect()
