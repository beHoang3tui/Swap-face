#!/usr/bin/env python3
"""
Simple Face Swap Tool
Dựa trên Deep-Live-Cam để swap face giữa 2 ảnh đơn giản
"""

import cv2
import insightface
import numpy as np
import os
import argparse
from typing import Any, Optional

class SimpleFaceSwap:
    def __init__(self, model_path: str = "models/inswapper_128_fp16.onnx"):
        """
        Khởi tạo face swap tool
        
        Args:
            model_path: Đường dẫn đến model insightface
        """
        self.face_analyser = None
        self.face_swapper = None
        self.model_path = model_path
        
    def get_face_analyser(self) -> Any:
        """Lấy face analyser từ insightface"""
        if self.face_analyser is None:
            self.face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        return self.face_analyser
    
    def get_face_swapper(self) -> Any:
        """Lấy face swapper model"""
        if self.face_swapper is None:
            if not os.path.exists(self.model_path):
                print(f"Model không tồn tại tại: {self.model_path}")
                print("Vui lòng tải model từ: https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx")
                return None
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
        """
        Thực hiện swap face
        
        Args:
            source_face: Khuôn mặt nguồn
            target_face: Khuôn mặt đích
            target_frame: Frame chứa khuôn mặt đích
            
        Returns:
            Frame đã được swap face
        """
        face_swapper = self.get_face_swapper()
        if face_swapper is None:
            print("Không thể load face swapper model!")
            return target_frame
            
        # Thực hiện face swap
        swapped_frame = face_swapper.get(
            target_frame, target_face, source_face, paste_back=True
        )
        
        return swapped_frame
    
    def process_image(self, source_path: str, target_path: str, output_path: str, many_faces: bool = False) -> bool:
        """
        Xử lý swap face cho ảnh
        
        Args:
            source_path: Đường dẫn ảnh nguồn
            target_path: Đường dẫn ảnh đích
            output_path: Đường dẫn ảnh output
            many_faces: Xử lý nhiều khuôn mặt
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            # Đọc ảnh
            source_frame = cv2.imread(source_path)
            target_frame = cv2.imread(target_path)
            
            if source_frame is None:
                print(f"Không thể đọc ảnh nguồn: {source_path}")
                return False
                
            if target_frame is None:
                print(f"Không thể đọc ảnh đích: {target_path}")
                return False
            
            # Lấy khuôn mặt từ ảnh nguồn
            source_face = self.get_one_face(source_frame)
            if source_face is None:
                print("Không tìm thấy khuôn mặt trong ảnh nguồn!")
                return False
            
            # Xử lý swap face
            if many_faces:
                # Xử lý nhiều khuôn mặt
                target_faces = self.get_many_faces(target_frame)
                if target_faces:
                    for target_face in target_faces:
                        target_frame = self.swap_face(source_face, target_face, target_frame)
                else:
                    print("Không tìm thấy khuôn mặt trong ảnh đích!")
                    return False
            else:
                # Xử lý một khuôn mặt
                target_face = self.get_one_face(target_frame)
                if target_face is None:
                    print("Không tìm thấy khuôn mặt trong ảnh đích!")
                    return False
                    
                target_frame = self.swap_face(source_face, target_face, target_frame)
            
            # Lưu ảnh kết quả
            cv2.imwrite(output_path, target_frame)
            print(f"Đã lưu ảnh kết quả tại: {output_path}")
            return True
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh: {str(e)}")
            return False
    
    def download_model(self):
        """Tải model nếu chưa có"""
        if not os.path.exists(self.model_path):
            print("Model chưa tồn tại. Vui lòng tải model từ:")
            print("https://huggingface.co/hacksider/deep-live-cam/blob/main/inswapper_128_fp16.onnx")
            print("Và đặt vào thư mục models/")

def main():
    parser = argparse.ArgumentParser(description='Simple Face Swap Tool')
    parser.add_argument('-s', '--source', required=True, help='Đường dẫn ảnh nguồn')
    parser.add_argument('-t', '--target', required=True, help='Đường dẫn ảnh đích')
    parser.add_argument('-o', '--output', required=True, help='Đường dẫn ảnh output')
    parser.add_argument('--many-faces', action='store_true', help='Xử lý nhiều khuôn mặt')
    parser.add_argument('--model-path', default='models/inswapper_128_fp16.onnx', help='Đường dẫn model')
    
    args = parser.parse_args()
    
    # Kiểm tra file tồn tại
    if not os.path.exists(args.source):
        print(f"Ảnh nguồn không tồn tại: {args.source}")
        return
    
    if not os.path.exists(args.target):
        print(f"Ảnh đích không tồn tại: {args.target}")
        return
    
    # Tạo thư mục output nếu chưa có
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Khởi tạo face swap tool
    face_swapper = SimpleFaceSwap(args.model_path)
    
    # Kiểm tra model
    if not os.path.exists(args.model_path):
        face_swapper.download_model()
        return
    
    # Thực hiện face swap
    success = face_swapper.process_image(
        args.source, 
        args.target, 
        args.output, 
        args.many_faces
    )
    
    if success:
        print("Face swap thành công!")
    else:
        print("Face swap thất bại!")

if __name__ == "__main__":
    main()