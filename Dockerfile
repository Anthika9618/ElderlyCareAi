
# ใช้ base image ที่มี Python
FROM python:3.9-slim

# ✅ เพิ่มคำสั่งนี้เพื่อติดตั้ง ps และ mosquitto-clients
RUN apt update && apt install -y procps mosquitto-clients

# ตั้ง working directory
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดเข้า container
COPY . .

# ติดตั้ง Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# รัน app.py เมื่อ container start
CMD ["python", "/app/app.py"]
