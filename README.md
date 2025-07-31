# Simple Face Swap Web App

Ứng dụng web đơn giản để swap face giữa 2 ảnh sử dụng AI.

## 🚀 Deploy lên Render.com (Khuyến nghị)

### Bước 1: Tạo tài khoản Render.com
1. Truy cập [render.com](https://render.com)
2. Đăng ký tài khoản miễn phí
3. Kết nối với GitHub

### Bước 2: Deploy
1. **Fork repository này về GitHub của bạn**
2. **Push code lên GitHub:**
   ```bash
   git add .
   git commit -m "Add Render.com deployment"
   git push
   ```

3. **Trên Render.com:**
   - Click "New +" → "Web Service"
   - Connect GitHub repository
   - Chọn repository
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python -c "import os, urllib.request; model_path = 'models/inswapper_128_fp16.onnx'; url = 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx'; os.makedirs('models', exist_ok=True); (not os.path.exists(model_path)) and (print('Downloading model...'), urllib.request.urlretrieve(url, model_path), print('Model downloaded!'))" && gunicorn -w 1 -b 0.0.0.0:$PORT --timeout 300 --max-requests 1000 --max-requests-jitter 50 app:app`
   - **Environment:** Python 3.9
   - Click "Create Web Service"

### Bước 3: Chờ deploy
- Render sẽ tự động build và deploy
- Model sẽ được tải tự động lần đầu
- URL sẽ được cung cấp sau khi deploy xong

## 🔧 Tính năng

- **Upload 2 ảnh** (source face + target face)
- **Swap face** với AI
- **Download kết quả**
- **Giao diện web** đẹp mắt
- **API endpoints** cho integration

## 📁 Cấu trúc

```
├── app.py              # Flask web app
├── requirements.txt    # Python dependencies
├── render.yaml        # Render.com config
├── Dockerfile         # Docker config (cho Railway)
└── README.md          # Hướng dẫn này
```

## 🛠️ Local Development

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Tải model (tự động)
python -c "import os, urllib.request; model_path = 'models/inswapper_128_fp16.onnx'; url = 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx'; os.makedirs('models', exist_ok=True); (not os.path.exists(model_path)) and (print('Downloading model...'), urllib.request.urlretrieve(url, model_path), print('Model downloaded!'))"

# Chạy app
python app.py
```

## 🌐 API Endpoints

- `GET /` - Giao diện web
- `POST /upload` - Upload ảnh
- `POST /swap` - Thực hiện face swap
- `GET /download/<filename>` - Download kết quả
- `GET /status` - Kiểm tra trạng thái
- `GET /health` - Health check

## 💡 Lưu ý

- **Render.com free tier** có 512MB RAM - đủ cho app này
- **Model tự động tải** lần đầu chạy
- **1 worker** để tiết kiệm memory
- **Timeout 5 phút** cho việc tải model

## 🆘 Troubleshooting

### Lỗi Memory
- Render.com free tier có giới hạn RAM
- App đã được tối ưu cho memory usage
- Nếu vẫn lỗi, cân nhắc upgrade plan

### Lỗi Model
- Model sẽ tự động tải lần đầu
- Kiểm tra logs để xem quá trình tải
- Có thể mất 5-10 phút lần đầu

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy:
1. Kiểm tra logs trên Render.com
2. Đảm bảo model đã được tải
3. Thử restart service 