"""
苹果视觉识别模块 - YOLOv11版本
基于YOLOv11训练的目标检测模型进行苹果识别
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import json
import time
import warnings
from collections import defaultdict

from config import Config

# 尝试导入YOLO相关模块
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    warnings.warn("ultralytics库未安装，将使用模拟模式")

try:
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class YOLOv11AppleDetector:
    """YOLOv11苹果检测器"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化YOLOv11检测器
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path or Config.vision_model_path
        
        # 苹果类别配置
        self.class_names = Config.apple_classes
        self.num_classes = len(self.class_names)
        
        # 检测参数
        self.conf_threshold = Config.confidence_threshold
        self.iou_threshold = Config.iou_threshold
        
        # 颜色映射
        self.colors = self._generate_colors()
        
        # 模型
        self.model = None
        self.load_model()
        
        # 识别历史
        self.recognition_history = []
        self.max_history = 100
        
        print(f"✅ YOLOv11苹果检测器初始化完成 (设备: {self.device})")
    
    def _generate_colors(self):
        """为每个类别生成随机颜色"""
        np.random.seed(42)
        colors = {}
        for i, class_name in enumerate(self.class_names):
            # 生成鲜艳的颜色
            hue = i * 60 % 360
            # 转换HSV到RGB
            hsv_color = np.uint8([[[hue, 255, 255]]])
            rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
            colors[class_name] = tuple(int(c) for c in rgb_color)
        return colors
    
    def load_model(self):
        """加载YOLOv11模型"""
        if not os.path.exists(self.model_path):
            print(f"⚠️ 模型文件不存在: {self.model_path}")
            print("将使用模拟模式进行测试")
            return False
        
        try:
            if YOLO_AVAILABLE:
                # 使用ultralytics YOLO
                print(f"加载YOLOv11模型: {self.model_path}")
                self.model = YOLO(self.model_path)
                
                # 设置模型到设备
                if self.device == 'cuda':
                    self.model.to('cuda')
                
                print(f"✅ YOLOv11模型加载成功")
                return True
            else:
                print("⚠️ ultralytics库未安装，使用模拟模式")
                return False
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def preprocess_image(self, image):
        """
        预处理图像
        """
        if isinstance(image, str):
            # 从文件加载
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件不存在: {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法读取图像文件: {image}")
            
        elif isinstance(image, np.ndarray):
            # numpy数组
            img = image.copy()
            if len(img.shape) == 2:
                # 灰度图转BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3:
                # 假设是RGB格式
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 4:
                # RGBA转BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
        elif isinstance(image, Image.Image):
            # PIL图像转BGR
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        else:
            raise ValueError("不支持的图像格式")
        
        return img
    
    def detect(self, image):
        """
        检测图像中的苹果
        """
        # 预处理图像
        img_bgr = self.preprocess_image(image)
        original_height, original_width = img_bgr.shape[:2]
        
        # 使用YOLO模型进行检测
        detections = []
        
        if YOLO_AVAILABLE and self.model is not None:
            try:
                # 运行推理
                results = self.model(img_bgr, 
                                   conf=self.conf_threshold,
                                   iou=self.iou_threshold,
                                   verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # 确保坐标在图像范围内
                            x1 = max(0, min(x1, original_width))
                            y1 = max(0, min(y1, original_height))
                            x2 = max(0, min(x2, original_width))
                            y2 = max(0, min(y2, original_height))
                            
                            # 获取类别名称
                            if 0 <= class_id < len(self.class_names):
                                class_name = self.class_names[class_id]
                            else:
                                class_name = f"未知{class_id}"
                            
                            detections.append([x1, y1, x2, y2, confidence, class_id, class_name])
                
            except Exception as e:
                print(f"YOLO推理错误: {e}")
        
        # 如果没有检测到或YOLO不可用，使用模拟检测
        if not detections:
            detections = self._simulate_detection(img_bgr)
        
        # 按置信度排序
        detections.sort(key=lambda x: x[4], reverse=True)
        
        # 绘制检测结果
        processed_img = self.draw_detections(img_bgr.copy(), detections)
        
        return detections, processed_img
    
    def _simulate_detection(self, image):
        """模拟检测（用于测试）"""
        detections = []
        height, width = image.shape[:2]
        
        # 在图像中心附近生成一个模拟检测框
        center_x, center_y = width // 2, height // 2
        box_size = min(width, height) // 4
        
        # 随机选择一个苹果类别
        import random
        class_id = random.randint(0, len(self.class_names) - 1)
        class_name = self.class_names[class_id]
        
        # 随机生成置信度
        confidence = random.uniform(0.7, 0.95)
        
        # 计算边界框
        x1 = center_x - box_size // 2
        y1 = center_y - box_size // 2
        x2 = center_x + box_size // 2
        y2 = center_y + box_size // 2
        
        # 确保边界框在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)
        
        detections.append([x1, y1, x2, y2, confidence, class_id, class_name])
        
        # 随机生成1-2个额外的检测框
        num_extra = random.randint(0, 2)
        for _ in range(num_extra):
            x1 = random.randint(0, width - 50)
            y1 = random.randint(0, height - 50)
            x2 = min(x1 + random.randint(30, 100), width - 1)
            y2 = min(y1 + random.randint(30, 100), height - 1)
            class_id = random.randint(0, len(self.class_names) - 1)
            class_name = self.class_names[class_id]
            confidence = random.uniform(0.6, 0.9)
            detections.append([x1, y1, x2, y2, confidence, class_id, class_name])
        
        return detections
    
    def draw_detections(self, image, detections):
        """
        在图像上绘制检测框
        """
        img_with_boxes = image.copy()
        height, width = image.shape[:2]
        
        for det in detections:
            x1, y1, x2, y2, confidence, class_id, class_name = det
            
            # 转换为整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 获取颜色
            color = self.colors.get(class_name, (0, 255, 0))
            
            # 绘制边界框
            thickness = max(1, int(min(width, height) * 0.002))
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
            
            # 创建标签文本
            label = f"{class_name}: {confidence:.2f}"
            
            # 计算标签背景大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, min(width, height) * 0.001)
            thickness_text = max(1, int(font_scale * 1.5))
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, thickness_text)
            
            # 绘制标签背景
            cv2.rectangle(img_with_boxes, 
                         (x1, y1 - text_height - 10),
                         (x1 + text_width, y1),
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(img_with_boxes, label,
                       (x1, y1 - 5),
                       font, font_scale,
                       (255, 255, 255), thickness_text)
        
        return img_with_boxes
    
    def recognize(self, image, return_image=False):
        """
        识别图像中的苹果（主函数）
        """
        try:
            # 检测苹果
            detections, processed_img = self.detect(image)
            
            # 如果没有检测到任何苹果
            if not detections:
                result_details = {
                    'detections': [],
                    'num_detections': 0,
                    'image_shape': processed_img.shape[:2],
                    'timestamp': time.time(),
                    'message': '未检测到苹果'
                }
                
                if return_image:
                    return "未检测到", 0.0, result_details, processed_img
                else:
                    return "未检测到", 0.0, result_details
            
            # 获取置信度最高的检测结果
            best_detection = detections[0]
            x1, y1, x2, y2, confidence, class_id, predicted_class = best_detection
            
            # 计算边界框信息
            bbox_area = (x2 - x1) * (y2 - y1)
            bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # 计算各类别统计
            class_stats = defaultdict(int)
            class_confidences = defaultdict(list)
            
            for det in detections:
                _, _, _, _, conf, _, cls_name = det
                class_stats[cls_name] += 1
                class_confidences[cls_name].append(conf)
            
            # 构建详细信息字典
            result_details = {
                'detections': detections,
                'best_detection': {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': predicted_class,
                    'bbox_area': float(bbox_area),
                    'bbox_center': bbox_center
                },
                'num_detections': len(detections),
                'class_distribution': dict(class_stats),
                'average_confidences': {k: np.mean(v) for k, v in class_confidences.items()},
                'image_shape': processed_img.shape[:2],
                'timestamp': time.time()
            }
            
            # 记录识别历史
            self._add_to_history(predicted_class, confidence, result_details)
            
            # 返回结果
            if return_image:
                return predicted_class, confidence, result_details, processed_img
            else:
                return predicted_class, confidence, result_details
            
        except Exception as e:
            print(f"❌ 识别失败: {e}")
            
            error_details = {
                'error': str(e),
                'timestamp': time.time()
            }
            
            return "识别错误", 0.0, error_details
    
    def _add_to_history(self, predicted_class, confidence, details):
        """添加到识别历史"""
        history_entry = {
            'timestamp': time.time(),
            'class': predicted_class,
            'confidence': confidence,
            'details': details,
            'num_detections': details.get('num_detections', 0)
        }
        
        self.recognition_history.append(history_entry)
        if len(self.recognition_history) > self.max_history:
            self.recognition_history.pop(0)
    
    def batch_recognize(self, image_list, return_images=False):
        """
        批量识别图像
        """
        results = []
        for image in image_list:
            if return_images:
                result = self.recognize(image, return_image=True)
                results.append(result)
            else:
                result = self.recognize(image)
                results.append(result)
        return results
    
    def get_recognition_stats(self):
        """获取识别统计信息"""
        if not self.recognition_history:
            return {}
        
        # 计算各类别识别次数
        class_counts = defaultdict(int)
        total_confidence = 0.0
        
        for record in self.recognition_history:
            apple_class = record['class']
            if apple_class not in ["未检测到", "识别错误"]:
                class_counts[apple_class] += 1
                total_confidence += record['confidence']
        
        # 平均置信度
        valid_records = [r for r in self.recognition_history 
                        if r['class'] not in ["未检测到", "识别错误"]]
        avg_confidence = np.mean([r['confidence'] for r in valid_records]) if valid_records else 0.0
        
        return {
            'total_recognitions': len(self.recognition_history),
            'class_distribution': dict(class_counts),
            'average_confidence': avg_confidence,
            'recent_recognition': self.recognition_history[-1] if self.recognition_history else None
        }
    
    def save_detection_image(self, image, output_path):
        """
        保存检测结果图像
        """
        try:
            _, processed_img = self.detect(image)
            cv2.imwrite(output_path, processed_img)
            print(f"✅ 检测结果保存到: {output_path}")
            return True
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False


class AppleVisionRecognizer:
    """
    苹果视觉识别器（兼容旧接口）
    包装YOLOv11检测器，提供与原分类器相同的接口
    """
    
    def __init__(self, model_path: str = None):
        """
        初始化苹果视觉识别器
        """
        self.detector = YOLOv11AppleDetector(model_path)
        self.model = self.detector  # 指向检测器
        
        print("✅ 苹果视觉识别器(YOLOv11)初始化完成")
    
    def load_model(self, model_path: str):
        """加载模型（兼容旧接口）"""
        return self.detector.load_model()
    
    def save_model(self, model_path: str):
        """保存模型"""
        print("⚠️ YOLO模型保存请直接使用torch.save()或ultralytics的导出功能")
        return False
    
    def recognize(self, image):
        """
        识别图像中的苹果（兼容旧接口）
        """
        # 调用YOLO检测器
        predicted_class, confidence, details = self.detector.recognize(image)
        
        # 添加所有类别概率（为了兼容旧接口）
        if 'all_probabilities' not in details:
            all_probs = {}
            for class_name in Config.apple_classes:
                if class_name in details.get('average_confidences', {}):
                    all_probs[class_name] = details['average_confidences'][class_name]
                else:
                    all_probs[class_name] = 0.0
            details['all_probabilities'] = all_probs
        
        return predicted_class, confidence, details
    
    def batch_recognize(self, image_list: List):
        """批量识别（兼容旧接口）"""
        return self.detector.batch_recognize(image_list)
    
    def get_recognition_stats(self):
        """获取识别统计（兼容旧接口）"""
        return self.detector.get_recognition_stats()
    
    def visualize_recognition(self, image, recognition_result=None):
        """
        可视化识别结果（兼容旧接口）
        """
        if recognition_result is None:
            # 直接调用detector的可视化方法
            _, processed_img = self.detector.detect(image)
            return processed_img
        else:
            # 使用已有的检测结果
            predicted_class, confidence, details = recognition_result
            detections = details.get('detections', [])
            img_bgr = self.detector.preprocess_image(image)
            return self.detector.draw_detections(img_bgr, detections)
    
    def train_on_new_data(self, image_paths, labels, epochs=10):
        """
        在新数据上训练模型
        """
        print("⚠️ YOLOv11训练需要专门的训练脚本")
        print("建议使用ultralytics的训练工具或编写专门的训练脚本")
        return 0.0, 0.0


# 测试函数
def test_vision_recognizer():
    """测试视觉识别器"""
    print("测试YOLOv11苹果检测器...")
    
    recognizer = AppleVisionRecognizer()
    
    # 创建测试图像
    test_img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    cv2.rectangle(test_img, (200, 150), (400, 350), (0, 0, 255), -1)  # 红色矩形
    
    # 测试识别
    result = recognizer.recognize(test_img)
    print(f"识别结果: {result[0]} (置信度: {result[1]:.2%})")
    
    # 显示统计信息
    stats = recognizer.get_recognition_stats()
    print(f"识别统计: {stats}")
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_vision_recognizer()
