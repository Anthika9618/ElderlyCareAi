import tensorflow as tf

model_path = "/mnt/c/ElderlyCareSystem/models/falldetect_bi_lstm_testmodel.h5"
print("กำลังโหลดโมเดลจาก:", model_path)

# load model โดยไม่ compile เพื่อลดปัญหาที่อาจเกิดขึ้นตอนโหลดโมเดลเก่า
model = tf.keras.models.load_model(model_path, compile=False)

print("โหลดโมเดลสำเร็จแล้ว")
