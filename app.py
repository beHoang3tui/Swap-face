#!/usr/bin/env python3
"""
Simple Face Swap Web App
Dựa trên simple_face_swap.py để tạo web interface
"""

import os
import cv2
import insightface
import numpy as np
import uuid
import logging
from flask import Flask, render_template_string, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
from typing import Any, Optional
import urllib.request

MODEL_PATH = 'models/inswapper_128_fp16.onnx'
MODEL_URL = 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx'

if not os.path.exists(MODEL_PATH):
    os.makedirs('models', exist_ok=True)
    print('Downloading model...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('Model downloaded!')

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Cấu hình upload
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Tạo thư mục nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Khởi tạo face swap tool
face_swapper = None

class SimpleFaceSwap:
    def __init__(self, model_path: str = "models/inswapper_128_fp16.onnx"):
        """Khởi tạo face swap tool"""
        self.face_analyser = None
        self.face_swapper = None
        self.model_path = model_path
        
    def get_face_analyser(self) -> Any:
        """Lấy face analyser từ insightface"""
        if self.face_analyser is None:
            logger.info("Khởi tạo face analyser...")
            self.face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        return self.face_analyser
    
    def get_face_swapper(self) -> Any:
        """Lấy face swapper model"""
        if self.face_swapper is None:
            if not os.path.exists(self.model_path):
                logger.error(f"Model không tồn tại tại: {self.model_path}")
                return None
            logger.info("Khởi tạo face swapper model...")
            self.face_swapper = insightface.model_zoo.get_model(
                self.model_path, providers=['CPUExecutionProvider']
            )
        return self.face_swapper
    
    def get_one_face(self, frame: np.ndarray) -> Optional[Any]:
        """Lấy khuôn mặt đầu tiên từ ảnh"""
        faces = self.get_face_analyser().get(frame)
        try:
            return min(faces, key=lambda x: x.bbox[0])
        except (ValueError, IndexError):
            return None
    
    def get_many_faces(self, frame: np.ndarray) -> Optional[list]:
        """Lấy tất cả khuôn mặt từ ảnh"""
        try:
            return self.get_face_analyser().get(frame)
        except IndexError:
            return None
    
    def swap_face(self, source_face: Any, target_face: Any, target_frame: np.ndarray) -> np.ndarray:
        """Thực hiện swap face"""
        face_swapper = self.get_face_swapper()
        if face_swapper is None:
            logger.error("Không thể load face swapper model!")
            return target_frame
            
        # Thực hiện face swap
        swapped_frame = face_swapper.get(
            target_frame, target_face, source_face, paste_back=True
        )
        
        return swapped_frame
    
    def process_image(self, source_path: str, target_path: str, output_path: str, many_faces: bool = False) -> bool:
        """Xử lý swap face cho ảnh"""
        try:
            logger.info(f"Đọc ảnh nguồn: {source_path}")
            source_frame = cv2.imread(source_path)
            
            logger.info(f"Đọc ảnh đích: {target_path}")
            target_frame = cv2.imread(target_path)
            
            if source_frame is None:
                logger.error(f"Không thể đọc ảnh nguồn: {source_path}")
                return False
                
            if target_frame is None:
                logger.error(f"Không thể đọc ảnh đích: {target_path}")
                return False
            
            # Lấy khuôn mặt từ ảnh nguồn
            logger.info("Phát hiện khuôn mặt trong ảnh nguồn...")
            source_face = self.get_one_face(source_frame)
            if source_face is None:
                logger.error("Không tìm thấy khuôn mặt trong ảnh nguồn!")
                return False
            
            # Xử lý swap face
            if many_faces:
                logger.info("Xử lý nhiều khuôn mặt...")
                target_faces = self.get_many_faces(target_frame)
                if target_faces:
                    logger.info(f"Tìm thấy {len(target_faces)} khuôn mặt trong ảnh đích")
                    for i, target_face in enumerate(target_faces):
                        logger.info(f"Swap face {i+1}/{len(target_faces)}")
                        target_frame = self.swap_face(source_face, target_face, target_frame)
                else:
                    logger.error("Không tìm thấy khuôn mặt trong ảnh đích!")
                    return False
            else:
                logger.info("Xử lý một khuôn mặt...")
                target_face = self.get_one_face(target_frame)
                if target_face is None:
                    logger.error("Không tìm thấy khuôn mặt trong ảnh đích!")
                    return False
                    
                target_frame = self.swap_face(source_face, target_face, target_frame)
            
            # Lưu ảnh kết quả
            logger.info(f"Lưu ảnh kết quả: {output_path}")
            cv2.imwrite(output_path, target_frame)
            logger.info("Face swap hoàn thành!")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            return False

def allowed_file(filename):
    """Kiểm tra file có được phép upload không"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_face_swapper():
    """Lấy face swapper instance"""
    global face_swapper
    if face_swapper is None:
        model_path = "models/inswapper_128_fp16.onnx"
        if not os.path.exists(model_path):
            logger.error("Model không tồn tại!")
            return None
        face_swapper = SimpleFaceSwap(model_path)
    return face_swapper

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Face Swap - Web App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s ease;
            background: #f8f9fa;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: #e3f2fd;
        }
        .preview-image {
            max-width: 200px;
            max-height: 200px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .btn-primary {
            background: linear-gradient(45deg, #667eea, #764ba2);
            border: none;
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        .result-image {
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        .status-badge {
            font-size: 0.8rem;
            padding: 5px 10px;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="main-container p-5">
                    <!-- Header -->
                    <div class="text-center mb-5">
                        <h1 class="display-4 fw-bold text-primary mb-3">
                            <i class="fas fa-magic me-3"></i>Simple Face Swap
                        </h1>
                        <p class="lead text-muted">Swap faces between images with AI-powered technology</p>
                    </div>

                    <!-- Upload Section -->
                    <div id="uploadSection">
                        <div class="row">
                            <div class="col-md-6 mb-4">
                                <h5><i class="fas fa-user me-2"></i>Source Face (Person to swap from)</h5>
                                <div class="upload-area" id="sourceUpload">
                                    <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                                    <p class="mb-2">Click or drag image here</p>
                                    <small class="text-muted">JPG, PNG, GIF, BMP (Max 16MB)</small>
                                    <input type="file" id="sourceFile" accept="image/*" style="display: none;">
                                </div>
                                <div id="sourcePreview" class="text-center mt-3" style="display: none;">
                                    <img id="sourceImage" class="preview-image">
                                </div>
                            </div>
                            <div class="col-md-6 mb-4">
                                <h5><i class="fas fa-users me-2"></i>Target Face (Person to swap to)</h5>
                                <div class="upload-area" id="targetUpload">
                                    <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
                                    <p class="mb-2">Click or drag image here</p>
                                    <small class="text-muted">JPG, PNG, GIF, BMP (Max 16MB)</small>
                                    <input type="file" id="targetFile" accept="image/*" style="display: none;">
                                </div>
                                <div id="targetPreview" class="text-center mt-3" style="display: none;">
                                    <img id="targetImage" class="preview-image">
                                </div>
                            </div>
                        </div>

                        <!-- Options -->
                        <div class="row mb-4">
                            <div class="col-12">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="manyFaces">
                                    <label class="form-check-label" for="manyFaces">
                                        <i class="fas fa-users me-2"></i>Process all faces in target image
                                    </label>
                                </div>
                            </div>
                        </div>

                        <!-- Upload Button -->
                        <div class="text-center">
                            <button class="btn btn-primary btn-lg" id="uploadBtn" disabled>
                                <i class="fas fa-upload me-2"></i>Upload Images
                            </button>
                        </div>
                    </div>

                    <!-- Processing Section -->
                    <div id="processingSection" style="display: none;">
                        <div class="text-center">
                            <i class="fas fa-spinner fa-spin fa-3x text-primary mb-3"></i>
                            <h4>Processing Face Swap...</h4>
                            <p class="text-muted">This may take a few moments</p>
                        </div>
                    </div>

                    <!-- Result Section -->
                    <div id="resultSection" style="display: none;">
                        <div class="text-center">
                            <h4 class="text-success mb-4">
                                <i class="fas fa-check-circle me-2"></i>Face Swap Completed!
                            </h4>
                            <div class="row">
                                <div class="col-md-6">
                                    <h6>Original</h6>
                                    <img id="originalImage" class="result-image mb-3">
                                </div>
                                <div class="col-md-6">
                                    <h6>Result</h6>
                                    <img id="resultImage" class="result-image mb-3">
                                </div>
                            </div>
                            <div class="mt-4">
                                <button class="btn btn-success me-2" id="downloadBtn">
                                    <i class="fas fa-download me-2"></i>Download Result
                                </button>
                                <button class="btn btn-outline-primary" id="newSwapBtn">
                                    <i class="fas fa-redo me-2"></i>New Swap
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- Error Section -->
                    <div id="errorSection" style="display: none;">
                        <div class="alert alert-danger text-center">
                            <i class="fas fa-exclamation-triangle fa-2x mb-3"></i>
                            <h5>Error</h5>
                            <p id="errorMessage"></p>
                            <button class="btn btn-outline-danger" id="tryAgainBtn">
                                <i class="fas fa-redo me-2"></i>Try Again
                            </button>
                        </div>
                    </div>

                    <!-- Status -->
                    <div class="text-center mt-4">
                        <span class="badge bg-secondary status-badge" id="statusBadge">
                            <i class="fas fa-circle me-1"></i>Ready
                        </span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Global variables
        let currentSession = null;
        let uploadedFiles = {
            source: null,
            target: null
        };

        // DOM elements
        const sourceUpload = document.getElementById('sourceUpload');
        const targetUpload = document.getElementById('targetUpload');
        const sourceFile = document.getElementById('sourceFile');
        const targetFile = document.getElementById('targetFile');
        const sourcePreview = document.getElementById('sourcePreview');
        const targetPreview = document.getElementById('targetPreview');
        const sourceImage = document.getElementById('sourceImage');
        const targetImage = document.getElementById('targetImage');
        const uploadBtn = document.getElementById('uploadBtn');
        const manyFaces = document.getElementById('manyFaces');
        const statusBadge = document.getElementById('statusBadge');

        // Sections
        const uploadSection = document.getElementById('uploadSection');
        const processingSection = document.getElementById('processingSection');
        const resultSection = document.getElementById('resultSection');
        const errorSection = document.getElementById('errorSection');

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            setupEventListeners();
            checkServerStatus();
        });

        function setupEventListeners() {
            // File upload handlers
            sourceUpload.addEventListener('click', () => sourceFile.click());
            targetUpload.addEventListener('click', () => targetFile.click());
            
            sourceFile.addEventListener('change', (e) => handleFileSelect(e, 'source'));
            targetFile.addEventListener('change', (e) => handleFileSelect(e, 'target'));

            // Drag and drop
            setupDragAndDrop(sourceUpload, sourceFile);
            setupDragAndDrop(targetUpload, targetFile);

            // Buttons
            uploadBtn.addEventListener('click', handleUpload);
            document.getElementById('downloadBtn').addEventListener('click', handleDownload);
            document.getElementById('newSwapBtn').addEventListener('click', resetApp);
            document.getElementById('tryAgainBtn').addEventListener('click', resetApp);
        }

        function setupDragAndDrop(uploadArea, fileInput) {
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    fileInput.dispatchEvent(new Event('change'));
                }
            });
        }

        function handleFileSelect(event, type) {
            const file = event.target.files[0];
            if (file) {
                uploadedFiles[type] = file;
                displayPreview(file, type);
                checkUploadButton();
            }
        }

        function displayPreview(file, type) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = type === 'source' ? sourceImage : targetImage;
                const preview = type === 'source' ? sourcePreview : targetPreview;
                
                img.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        function checkUploadButton() {
            uploadBtn.disabled = !(uploadedFiles.source && uploadedFiles.target);
        }

        async function handleUpload() {
            if (!uploadedFiles.source || !uploadedFiles.target) return;

            const formData = new FormData();
            formData.append('source', uploadedFiles.source);
            formData.append('target', uploadedFiles.target);

            try {
                updateStatus('Uploading files...', 'warning');

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    currentSession = data;
                    await handleSwap();
                } else {
                    throw new Error(data.error);
                }
            } catch (error) {
                showError(error.message);
            }
        }

        async function handleSwap() {
            try {
                showSection(processingSection);
                updateStatus('Processing face swap...', 'warning');

                const response = await fetch('/swap', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        session_id: currentSession.session_id,
                        source_path: currentSession.source_path,
                        target_path: currentSession.target_path,
                        many_faces: manyFaces.checked
                    })
                });

                const data = await response.json();

                if (data.success) {
                    showResult(data);
                } else {
                    throw new Error(data.error);
                }
            } catch (error) {
                showError(error.message);
            }
        }

        function showResult(data) {
            updateStatus('Completed!', 'success');
            
            // Display images
            document.getElementById('originalImage').src = URL.createObjectURL(uploadedFiles.target);
            document.getElementById('resultImage').src = data.result_url + '?t=' + new Date().getTime();
            
            showSection(resultSection);
        }

        function handleDownload() {
            if (currentSession) {
                const link = document.createElement('a');
                link.href = `/download/result_${currentSession.session_id}.jpg`;
                link.download = 'face_swap_result.jpg';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        }

        function showError(message) {
            document.getElementById('errorMessage').textContent = message;
            showSection(errorSection);
            updateStatus('Error', 'danger');
        }

        function resetApp() {
            uploadedFiles = { source: null, target: null };
            currentSession = null;
            
            // Reset UI
            sourcePreview.style.display = 'none';
            targetPreview.style.display = 'none';
            sourceFile.value = '';
            targetFile.value = '';
            manyFaces.checked = false;
            uploadBtn.disabled = true;
            
            updateStatus('Ready', 'secondary');
            showSection(uploadSection);
        }

        function showSection(section) {
            [uploadSection, processingSection, resultSection, errorSection].forEach(s => {
                s.style.display = 'none';
            });
            section.style.display = 'block';
        }

        function updateStatus(message, type) {
            statusBadge.className = `badge bg-${type} status-badge`;
            statusBadge.innerHTML = `<i class="fas fa-circle me-1"></i>${message}`;
        }

        async function checkServerStatus() {
            try {
                const response = await fetch('/status');
                const data = await response.json();
                
                if (!data.model_loaded) {
                    updateStatus('Model not loaded', 'danger');
                } else {
                    updateStatus('Ready', 'secondary');
                }
            } catch (error) {
                updateStatus('Server error', 'danger');
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Trang chủ"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    """API upload file"""
    try:
        if 'source' not in request.files or 'target' not in request.files:
            return jsonify({'error': 'Thiếu file source hoặc target'}), 400
        
        source_file = request.files['source']
        target_file = request.files['target']
        
        if source_file.filename == '' or target_file.filename == '':
            return jsonify({'error': 'Chưa chọn file'}), 400
        
        if not (allowed_file(source_file.filename) and allowed_file(target_file.filename)):
            return jsonify({'error': 'Định dạng file không được hỗ trợ'}), 400
        
        # Tạo unique ID cho session
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        # Lưu files
        source_path = os.path.join(session_folder, secure_filename(source_file.filename))
        target_path = os.path.join(session_folder, secure_filename(target_file.filename))
        
        source_file.save(source_path)
        target_file.save(target_path)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'source_path': source_path,
            'target_path': target_path
        })
        
    except Exception as e:
        logger.error(f"Lỗi upload: {str(e)}")
        return jsonify({'error': f'Lỗi upload: {str(e)}'}), 500

@app.route('/swap', methods=['POST'])
def swap_faces():
    """API swap faces"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        source_path = data.get('source_path')
        target_path = data.get('target_path')
        many_faces = data.get('many_faces', False)
        
        if not all([session_id, source_path, target_path]):
            return jsonify({'error': 'Thiếu thông tin cần thiết'}), 400
        
        # Kiểm tra files tồn tại
        if not os.path.exists(source_path) or not os.path.exists(target_path):
            return jsonify({'error': 'Files không tồn tại'}), 400
        
        # Lấy face swapper
        swapper = get_face_swapper()
        if swapper is None:
            return jsonify({'error': 'Không thể khởi tạo face swapper. Vui lòng kiểm tra model.'}), 500
        
        # Tạo output path
        output_filename = f"result_{session_id}.jpg"
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        
        # Thực hiện face swap
        success = swapper.process_image(source_path, target_path, output_path, many_faces)
        
        if success:
            return jsonify({
                'success': True,
                'result_path': output_path,
                'result_url': url_for('download_result', filename=output_filename)
            })
        else:
            return jsonify({'error': 'Face swap thất bại. Vui lòng kiểm tra ảnh có khuôn mặt rõ ràng.'}), 500
            
    except Exception as e:
        logger.error(f"Lỗi swap faces: {str(e)}")
        return jsonify({'error': f'Lỗi xử lý: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_result(filename):
    """Download kết quả"""
    try:
        return send_file(os.path.join(RESULT_FOLDER, filename), as_attachment=True)
    except Exception as e:
        logger.error(f"Lỗi download: {str(e)}")
        return jsonify({'error': 'File không tồn tại'}), 404

@app.route('/status')
def status():
    """API kiểm tra trạng thái"""
    model_path = "models/inswapper_128_fp16.onnx"
    model_exists = os.path.exists(model_path)
    
    return jsonify({
        'status': 'running',
        'model_loaded': model_exists,
        'model_path': model_path if model_exists else None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Kiểm tra model trước khi chạy
    model_path = "models/inswapper_128_fp16.onnx"
    if not os.path.exists(model_path):
        logger.error("❌ Model không tồn tại!")
        logger.info("💡 Vui lòng tải model từ: https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx")
        logger.info("💡 Và đặt vào thư mục models/")
        exit(1)
    
    logger.info("🚀 Khởi động Simple Face Swap Web App...")
    logger.info(f"📁 Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"📁 Result folder: {RESULT_FOLDER}")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Mở trình duyệt tại: http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=False) 