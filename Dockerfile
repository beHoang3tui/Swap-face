# Sử dụng Python slim để giảm dung lượng image
FROM python:3.9-slim

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV, insightface và build tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements trước để tận dụng cache layer
COPY requirements.txt .

# Cài đặt Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào image
COPY . .

# Tạo thư mục models nếu chưa có
RUN mkdir -p models uploads results

# Expose port (Railway, Render, ... sẽ tự map port này)
EXPOSE 5000

# Lệnh khởi động: tải model nếu chưa có, sau đó chạy gunicorn với tối ưu memory
CMD python -c "import os, urllib.request; model_path = 'models/inswapper_128_fp16.onnx'; url = 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx'; os.makedirs('models', exist_ok=True); (not os.path.exists(model_path)) and (print('Downloading model...'), urllib.request.urlretrieve(url, model_path), print('Model downloaded!'))" && gunicorn -w 1 -b 0.0.0.0:5000 --timeout 300 --max-requests 1000 --max-requests-jitter 50 app:app 