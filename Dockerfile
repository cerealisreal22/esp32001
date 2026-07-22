FROM python:3.11-slim

# ติดตั้ง Library ระบบ Linux ที่ MediaPipe และ OpenCV ต้องใช้
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง Python Packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้โค้ดทั้งหมดเข้า Container
COPY . .

EXPOSE 10000

# คำสั่งรันเซิร์ฟเวอร์
CMD ["python", "server.py"]
