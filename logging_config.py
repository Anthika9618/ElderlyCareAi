import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # กำหนดระดับ log ขั้นต่ำ

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    # log ลงไฟล์ หมุนไฟล์เมื่อขนาดเกิน 5MB เก็บไว้ 5 ไฟล์
    file_handler = RotatingFileHandler('app.log', maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # log ลง console ด้วย
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
