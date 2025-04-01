import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy import stats
from skimage.feature import graycomatrix, graycoprops
from math import pi, cos, sin, radians, sqrt
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading

class ParathyroidTumorAnalyzer:
    def __init__(self, image_dir, json_dir, output_dir, px_per_mm=19, progress_callback=None):
        """
        初始化分析器
        
        Args:
            image_dir (str): 原始影像的目錄
            json_dir (str): labelme標註JSON檔案的目錄
            output_dir (str): 結果輸出目錄
            px_per_mm (float): 像素到毫米的轉換比例，默認1mm=19px
            progress_callback (callable, optional): 用於更新進度的回調函數
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.output_dir = output_dir
        self.px_per_mm = px_per_mm  # 像素到毫米的轉換比例
        self.progress_callback = progress_callback
        
        # 創建輸出目錄（如果不存在）
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, "visualizations")):
            os.makedirs(os.path.join(output_dir, "visualizations"))
        
        # 用於存儲所有結果的列表
        self.results = []
        self.processed_images = []
        
    def analyze_all_images(self):
        """分析所有的標註影像"""
        # 取得所有JSON檔案
        json_files = glob(os.path.join(self.json_dir, "*.json"))
        total_files = len(json_files)
        
        # 預先載入所有JSON檔案，避免重複讀取
        json_cache = {}
        for json_file in json_files:
            base_name = os.path.basename(json_file).replace(".json", "")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_cache[base_name] = json.load(f)
            except Exception as e:
                print(f"Error loading JSON file {json_file}: {str(e)}")
                if self.progress_callback:
                    self.progress_callback(0, total_files, f"Error loading JSON: {base_name}")
        
        for idx, json_file in enumerate(json_files):
            # 從JSON檔案名稱中獲取對應的影像檔案名稱
            base_name = os.path.basename(json_file).replace(".json", "")
            image_file = os.path.join(self.image_dir, f"{base_name}.jpg")
            
            # 如果找不到JPG格式，嘗試PNG格式
            if not os.path.exists(image_file):
                image_file = os.path.join(self.image_dir, f"{base_name}.png")
            
            if os.path.exists(image_file) and base_name in json_cache:
                if self.progress_callback:
                    self.progress_callback(idx, total_files, f"Processing image: {base_name}")
                self.analyze_image(image_file, json_cache[base_name], base_name)
                self.processed_images.append(os.path.join(self.output_dir, "visualizations", f"{base_name}_analysis.png"))
            else:
                if self.progress_callback:
                    self.progress_callback(idx, total_files, f"Image file not found: {base_name}")
        
        # 現在只在這裡保存CSV結果，避免重複
        self.save_results_to_csv()
        
    def calculate_glcm_features(self, roi_gray, mask):
        """計算灰度共生矩陣(GLCM)的紋理特徵"""
        try:
            # 確保ROI區域足夠大以計算GLCM
            if np.count_nonzero(mask) > 25:  # 至少需要5x5的區域
                # 僅提取包含腫瘤的區域的最小矩形
                y_indices, x_indices = np.where(mask > 0)
                top, bottom = np.min(y_indices), np.max(y_indices)
                left, right = np.min(x_indices), np.max(x_indices)
                
                # 確保提取區域至少有2x2的大小，否則GLCM計算會失敗
                if right - left < 1 or bottom - top < 1:
                    raise ValueError("ROI too small for GLCM calculation")
                
                # 提取ROI
                roi_small = roi_gray[top:bottom+1, left:right+1]
                mask_small = mask[top:bottom+1, left:right+1]
                
                # 創建一個只包含腫瘤區域的圖像
                roi_tumor_only = np.zeros_like(roi_small)
                roi_tumor_only[mask_small > 0] = roi_small[mask_small > 0]
                
                # 縮放灰度範圍到較少的級別以避免稀疏GLCM
                levels = 8
                roi_rescaled = (roi_tumor_only // (256 // levels)).astype(np.uint8)
                
                # 移除所有零值像素（不屬於腫瘤區域）
                non_zero_mask = roi_rescaled > 0
                
                # 如果非零部分太小，則無法計算GLCM
                if np.count_nonzero(non_zero_mask) < 4:  # 至少需要2x2的有效區域
                    raise ValueError("Effective ROI too small for GLCM calculation")
                
                # 計算GLCM (距離=1, 方向=[0, 45, 90, 135]度)
                # 注意：不使用mask參數，因為我們已經提取了只包含腫瘤的區域
                glcm = graycomatrix(roi_rescaled, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                   levels=levels, symmetric=True, normed=True)
                
                # 計算GLCM屬性
                contrast = np.mean(graycoprops(glcm, 'contrast')[0])
                homogeneity = np.mean(graycoprops(glcm, 'homogeneity')[0])
                energy = np.mean(graycoprops(glcm, 'energy')[0])
                correlation = np.mean(graycoprops(glcm, 'correlation')[0])
                dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity')[0])
                ASM = np.mean(graycoprops(glcm, 'ASM')[0])
                
                # 計算熵
                # 首先將GLCM平坦化並移除零元素
                glcm_flat = glcm.flatten()
                glcm_flat = glcm_flat[glcm_flat > 0]
                entropy = -np.sum(glcm_flat * np.log2(glcm_flat)) if len(glcm_flat) > 0 else np.nan
            else:
                raise ValueError("ROI too small for GLCM calculation")
        except Exception as e:
            print(f"Error in GLCM calculation: {e}")
            contrast = homogeneity = energy = correlation = dissimilarity = ASM = entropy = np.nan
            
        return {
            'GLCM_Contrast': contrast,
            'GLCM_Homogeneity': homogeneity,
            'GLCM_Energy': energy,
            'GLCM_Correlation': correlation, 
            'GLCM_Dissimilarity': dissimilarity,
            'GLCM_ASM': ASM,
            'GLCM_Entropy': entropy
        }
        
    def calculate_ellipse_features(self, points):
        """計算擬合橢圓的特徵"""
        if len(points) < 5:  # 至少需要5個點才能擬合橢圓
            return {
                'Ellipse_Area': np.nan,
                'Ellipse_Perimeter': np.nan,
                'Ellipse_MajorAxis': np.nan,
                'Ellipse_MajorAxis_mm': np.nan,
                'Ellipse_MinorAxis': np.nan,
                'Ellipse_MajorAxis_Angle': np.nan,
                'Ellipse_MinorAxis_Angle': np.nan
            }
            
        try:
            # 使用OpenCV的擬合橢圓函數
            ellipse = cv2.fitEllipse(points)
            center, axes, angle = ellipse
            
            # 獲取長軸和短軸
            a, b = axes[0] / 2, axes[1] / 2  # 除以2轉為半長軸和半短軸
            major_axis = max(a, b) * 2
            minor_axis = min(a, b) * 2
            
            # 計算橢圓面積
            ellipse_area = pi * a * b
            
            # 使用Ramanujan公式近似計算橢圓周長
            h = ((a - b) ** 2) / ((a + b) ** 2)
            ellipse_perimeter = pi * (a + b) * (1 + 3*h / (10 + sqrt(4 - 3*h)))
            
            # 確定長軸角度 - OpenCV返回的是水平方向逆時針到主軸的角度
            if a >= b:  # 如果水平軸是長軸
                major_angle = angle
            else:  # 如果垂直軸是長軸
                major_angle = angle + 90
                
            # 規範化角度到0-180度
            major_angle = major_angle % 180
            
            # 短軸角度總是與長軸垂直
            minor_angle = (major_angle + 90) % 180

            px_to_mm = 1.0 / self.px_per_mm  # 1mm=19px
            
            major_axis_mm = major_axis * px_to_mm
            
            return {
                'Ellipse_Area': ellipse_area,
                'Ellipse_Perimeter': ellipse_perimeter,
                'Ellipse_MajorAxis': major_axis,
                'Ellipse_MajorAxis_mm': major_axis_mm,
                'Ellipse_MinorAxis': minor_axis,
                'Ellipse_MajorAxis_Angle': major_angle,
                'Ellipse_MinorAxis_Angle': minor_angle
            }
        except Exception as exc:
            print(f"Error calculating ellipse features: {str(exc)}")
            return {
                'Ellipse_Area': np.nan,
                'Ellipse_Perimeter': np.nan,
                'Ellipse_MajorAxis': np.nan,
                'Ellipse_MajorAxis_mm': np.nan,
                'Ellipse_MinorAxis': np.nan,
                'Ellipse_MajorAxis_Angle': np.nan,
                'Ellipse_MinorAxis_Angle': np.nan
            }
            
    def calculate_shape_features(self, mask, points):
        """計算形狀特徵"""
        area = cv2.countNonZero(mask)
        perimeter = cv2.arcLength(points, True)
        
        # 計算圓形度 (4π * Area / Perimeter2)
        circularity = 4 * pi * area / (perimeter ** 2) if perimeter > 0 else np.nan
        
        # 計算凸包
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, True)
        
        # 計算凸性 (Convexity) = 凸包周長 / 實際周長
        convexity = hull_perimeter / perimeter if perimeter > 0 else np.nan
        
        # 計算凹陷指數 (Solidity) = 區域面積 / 凸包面積
        solidity = area / hull_area if hull_area > 0 else np.nan
        solidity = 1.0 if solidity > 1 else solidity
        
        # 計算邊界不規則度 (Irregularity Index) = 實際周長 / 凸包周長
        irregularity = perimeter / hull_perimeter if hull_perimeter > 0 else np.nan
        
        # 計算長寬比 (使用外接矩形)
        x, y, w, h = cv2.boundingRect(points)
        if w >= h:
            aspect_ratio = w / h
        elif w < h:
            aspect_ratio = h / w
        else:
            aspect_ratio = np.nan
        
        # 計算費內特直徑 (Feret's Diameter) - 對象兩點之間的最大距離
        # 計算所有點對之間的最大距離
        max_distance = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.sqrt((points[i][0][0] - points[j][0][0])**2 + (points[i][0][1] - points[j][0][1])**2)
                if dist > max_distance:
                    max_distance = dist
        
        # 補充區域分數 (Area Fraction) - 區域面積與外接矩形面積的比率
        area_fraction = area / (w * h) if (w * h) > 0 else np.nan
        
        return {
            'Circularity': circularity,
            'Aspect_Ratio': aspect_ratio,
            'Irregularity_Index': irregularity,
            'Convexity': convexity,
            'Solidity': solidity,
            'Ferets_Diameter': max_distance,
            'Area_Fraction': area_fraction
        }
        
    def analyze_image(self, image_path, annotation_data, base_name):
        """
        分析單一影像及其標註
        
        Args:
            image_path (str): 影像檔案路徑
            annotation_data (dict): 解析後的標註數據
            base_name (str): 檔案基本名稱（不含副檔名）
        """
        # 讀取影像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Cannot read image: {image_path}")
            return
            
        # 將BGR轉為RGB用於顯示
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 將彩色影像轉為灰階
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 創建可視化結果
        visualization = image_rgb.copy()
        
        # 為右上角文字信息預留空間
        info_texts = []
        
        # 處理每個標註的區域
        for idx, shape in enumerate(annotation_data['shapes']):
            if shape['shape_type'] == 'polygon':
                # 獲取多邊形的點
                points = np.array(shape['points'], dtype=np.int32)
                
                # 創建遮罩
                mask = np.zeros(gray_image.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                
                # 計算區域的面積（像素數）
                area_pixels = cv2.countNonZero(mask)
                
                # 計算灰階區域的統計數據
                roi_gray = cv2.bitwise_and(gray_image, mask)
                
                # 只考慮遮罩內的像素點
                roi_pixels = roi_gray[mask > 0]
                
                if len(roi_pixels) > 0:
                    # 二值化處理（使用Otsu's方法自動選擇閾值）
                    _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # 計算二值化後的區域內的統計數據
                    binary_roi = binary[mask > 0]
                    
                    # 1. 基本亮度統計
                    mean_intensity = np.mean(roi_pixels)
                    median_intensity = np.median(roi_pixels)
                    max_intensity = np.max(roi_pixels)
                    min_intensity = np.min(roi_pixels)
                    std_intensity = np.std(roi_pixels)
                    binary_mean = np.mean(binary_roi)
                    
                    # 2. 計算高階統計量
                    skewness = stats.skew(roi_pixels) if len(roi_pixels) > 2 else np.nan
                    kurtosis = stats.kurtosis(roi_pixels) if len(roi_pixels) > 3 else np.nan
                    
                    # 3. 將多邊形點轉換為可用於輪廓分析的格式
                    contour_points = points.reshape(-1, 1, 2)
                    
                    # 4. 計算形狀特徵
                    shape_features = self.calculate_shape_features(mask, contour_points)
                    
                    # 5. 計算橢圓特徵
                    ellipse_features = self.calculate_ellipse_features(contour_points)
                    
                    # 6. 計算紋理特徵 (GLCM)
                    glcm_features = self.calculate_glcm_features(gray_image, mask)
                    
                    # 7. 將數值標準化 (Z-分數)
                    # if len(roi_pixels) > 1:
                    #     z_scores = (roi_pixels - np.mean(roi_pixels)) / np.std(roi_pixels)
                    #     normalized_intensity = np.mean(z_scores)
                    # else:
                    #     normalized_intensity = np.nan
                    
                    # 計算多邊形的周長和面積
                    perimeter = cv2.arcLength(contour_points, True)
                    area_pixels = cv2.countNonZero(mask)
                    
                    # 轉換為毫米單位 (使用用戶自定義的轉換比例)
                    px_to_mm = 1.0 / self.px_per_mm  # 例如，如果1mm=19px
                    perimeter_mm = perimeter * px_to_mm
                    # 面積轉換需要平方關係
                    area_mm = area_pixels * (px_to_mm ** 2)
                    
                    # 儲存結果
                    tumor_id = f"{base_name}_tumor_{idx+1}"
                    result_dict = {
                        'Image': base_name,
                        'Tumor_ID': tumor_id,
                        'Area_Pixels': area_pixels,
                        'Area_mm2': area_mm,
                        'Perimeter_Pixels': perimeter,
                        'Perimeter_mm': perimeter_mm,
                        # 基本亮度統計
                        'Mean_Intensity': mean_intensity,
                        'Median_Intensity': median_intensity,
                        'Min_Intensity': min_intensity,
                        'Max_Intensity': max_intensity,
                        'Std_Intensity': std_intensity,
                        'Binary_Mean_Intensity': binary_mean,
                        # 高階統計量
                        'Skewness': skewness,
                        'Kurtosis': kurtosis,
                        # 'Normalized_Intensity': normalized_intensity,
                    }
                    
                    # 合併所有特徵
                    result_dict.update(shape_features)
                    result_dict.update(ellipse_features)
                    result_dict.update(glcm_features)
                    
                    self.results.append(result_dict)
                    
                    # 在可視化影像上畫出區域輪廓
                    cv2.polylines(visualization, [contour_points], True, (255, 0, 0), 2)
                    
                    # 在腫瘤中心顯示ID
                    centroid = np.mean(points, axis=0, dtype=np.int32)
                    cv2.putText(visualization, f"{idx+1}", 
                               (centroid[0], centroid[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
                    # 收集這個腫瘤的信息，稍後統一顯示在右上角
                    angle_info = "N/A"
                    major_axis_mm = "N/A"
                    if not np.isnan(ellipse_features['Ellipse_MajorAxis']):
                        # 直接使用橢圓特徵計算中的角度，確保一致性
                        major_angle = ellipse_features['Ellipse_MajorAxis_Angle']
                        angle_info = f"Angle: {major_angle:.1f} deg"
                        major_axis_mm = f"MajorAxis: {ellipse_features['Ellipse_MajorAxis_mm']:.2f} mm"
                    
                    # 收集完整的信息
                    info_text = {
                        'id': idx+1,
                        'text': [
                            f"ID: {idx+1}",
                            f"Area: {area_mm:.2f} mm2",  # 顯示毫米單位的面積
                            f"Perimeter: {perimeter_mm:.2f} mm",  # 顯示毫米單位的周長
                            major_axis_mm,
                            angle_info
                        ]
                    }
                    info_texts.append(info_text)
                    
                    # 擬合橢圓並繪製
                    if not np.isnan(ellipse_features['Ellipse_MajorAxis']):
                        try:
                            ellipse = cv2.fitEllipse(contour_points)
                            center, axes, angle = ellipse
                            
                            # 繪製橢圓輪廓
                            cv2.ellipse(visualization, ellipse, (0, 255, 0), 2)
                            
                            # 確定長軸和短軸
                            width, height = axes
                            if width > height:
                                # 寬度大於高度，寬度是長軸
                                major_axis_length = width / 2
                                minor_axis_length = height / 2
                                # 角度已經是相對於長軸的，不需調整
                                major_angle_rad = radians(angle)
                                display_angle = angle
                            else:
                                # 高度大於寬度，高度是長軸
                                major_axis_length = height / 2
                                minor_axis_length = width / 2
                                # 角度需要調整90度，使其相對於長軸
                                major_angle_rad = radians(angle + 90)
                                display_angle = angle + 90
                                # 保持角度在0-180範圍內
                                if display_angle >= 180:
                                    display_angle -= 180
                            
                            # 計算長軸的端點
                            x1 = int(center[0] - major_axis_length * cos(major_angle_rad))
                            y1 = int(center[1] - major_axis_length * sin(major_angle_rad))
                            x2 = int(center[0] + major_axis_length * cos(major_angle_rad))
                            y2 = int(center[1] + major_axis_length * sin(major_angle_rad))
                            
                            # 繪製長軸
                            cv2.line(visualization, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            # 短軸的角度總是比長軸的角度多90度
                            minor_angle_rad = major_angle_rad + pi/2
                            
                            # 計算短軸的端點
                            x3 = int(center[0] - minor_axis_length * cos(minor_angle_rad))
                            y3 = int(center[1] - minor_axis_length * sin(minor_angle_rad))
                            x4 = int(center[0] + minor_axis_length * cos(minor_angle_rad))
                            y4 = int(center[1] + minor_axis_length * sin(minor_angle_rad))
                            
                            # 繪製短軸
                            cv2.line(visualization, (x3, y3), (x4, y4), (255, 255, 0), 2)
                            
                            # 更新角度信息
                            angle_info = f"Angle: {display_angle:.1f} deg"
                            
                            # 繪製水平參考線以顯示夾角（用點線模擬虛線效果）
                            line_length = max(major_axis_length, 100)
                            # 用多段短線模擬虛線效果
                            dash_length = 10
                            gap_length = 5
                            start_x = int(center[0] - line_length)
                            end_x = int(center[0] + line_length)
                            y = int(center[1])
                            
                            for x in range(start_x, end_x, dash_length + gap_length):
                                line_end = min(x + dash_length, end_x)
                                cv2.line(visualization, (x, y), (line_end, y), (255, 0, 255), 1)
                            
                        except Exception as e:
                            print(f"Error drawing ellipse: {e}")
        
        # 在右上角顯示所有腫瘤的信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        padding = 10
        
        for info in info_texts:
            # 計算這個信息塊的高度
            block_height = len(info['text']) * 30
            
            # 在右上角顯示每一行信息
            for i, line in enumerate(info['text']):
                if line:  # 只顯示非空行
                    cv2.putText(visualization, line, 
                              (visualization.shape[1] - 300, y_offset + i * 30), 
                              font, 0.7, (255, 255, 255), 2)
            
            # 更新下一個信息塊的y起始位置
            y_offset += block_height + padding
        
        # 保存可視化結果
        vis_path = os.path.join(self.output_dir, "visualizations", f"{base_name}_analysis.png")
        plt.figure(figsize=(12, 8))
        plt.imshow(visualization)
        plt.title(f"Parathyroid Tumor Analysis - {base_name}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(vis_path, dpi=300)
        plt.close()
        
    def save_results_to_csv(self):
        """將所有分析結果保存到CSV檔案"""
        if self.results:
            csv_path = os.path.join(self.output_dir, "parathyroid_analysis_results.csv")
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False)
            print(f"Analysis results saved to: {csv_path}")
            return csv_path
        else:
            print("No analyzable images found")
            return None


class ParathyroidAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Parathyroid Analyzer")
        self.master.geometry("900x700")
        # 允許調整窗口大小
        
        # 拖拽相關變量
        self.drag_data = {"x": 0, "y": 0, "dragging": False}
        self.image_item = None  # 保存Canvas上的圖片項目ID
        self.can_drag = False  # 標記是否可以拖動 (只有在放大時才能拖動)
        
        # 設定默認目錄和轉換比例
        self.image_dir = tk.StringVar(value=os.path.join(os.getcwd(), "images"))
        self.json_dir = tk.StringVar(value=os.path.join(os.getcwd(), "annotations"))
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "results"))
        self.px_to_mm_ratio = tk.StringVar(value="19")  # 預設1mm=19px
        
        self.create_widgets()
        self.center_window()
        
        # 分析結果
        self.results_csv = None
        self.current_image_index = 0
        self.processed_images = []
        self.current_zoom = 1.0
        self.original_image = None
        
    def center_window(self):
        self.master.update_idletasks()
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    def create_widgets(self):
        # 創建主框架
        main_frame = ttk.Frame(self.master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左側面板 - 設置和控制
        left_frame = ttk.Frame(main_frame, padding="5", borderwidth=2, relief="groove")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(0, 10))
        
        # 右側面板 - 顯示區域
        right_frame = ttk.Frame(main_frame, padding="5")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        #---------- 左側面板內容 ----------#
        # 標題
        title_label = ttk.Label(left_frame, text="Parathyroid Analyzer", font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 15))
        
        # 路徑設置區域
        path_frame = ttk.LabelFrame(left_frame, text="Path Settings", padding=10)
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 影像目錄
        ttk.Label(path_frame, text="Image Directory:").pack(anchor=tk.W)
        image_path_frame = ttk.Frame(path_frame)
        image_path_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Entry(image_path_frame, textvariable=self.image_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(image_path_frame, text="Browse", command=self.browse_image_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 標註目錄
        ttk.Label(path_frame, text="Annotation Directory:").pack(anchor=tk.W)
        json_path_frame = ttk.Frame(path_frame)
        json_path_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Entry(json_path_frame, textvariable=self.json_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(json_path_frame, text="Browse", command=self.browse_json_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 輸出目錄
        ttk.Label(path_frame, text="Output Directory:").pack(anchor=tk.W)
        output_path_frame = ttk.Frame(path_frame)
        output_path_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Entry(output_path_frame, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_path_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 像素到毫米的轉換比例
        ttk.Label(path_frame, text="Conversion Ratio (1mm = ? px):").pack(anchor=tk.W)
        ratio_frame = ttk.Frame(path_frame)
        ratio_frame.pack(fill=tk.X)
        ttk.Entry(ratio_frame, textvariable=self.px_to_mm_ratio, width=8).pack(side=tk.LEFT)
        ttk.Label(ratio_frame, text="(Default: 19px)").pack(side=tk.LEFT, padx=5)
        
        # 分析按鈕和進度條
        control_frame = ttk.LabelFrame(left_frame, text="Analysis Control", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.analyze_button = ttk.Button(control_frame, text="Start Analysis", command=self.start_analysis)
        self.analyze_button.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 5))
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_var, wraplength=200)
        self.status_label.pack(fill=tk.X)
        
        # 結果操作區域
        results_frame = ttk.LabelFrame(left_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.open_csv_button = ttk.Button(results_frame, text="Open CSV Results", command=self.open_csv, state=tk.DISABLED)
        self.open_csv_button.pack(fill=tk.X, pady=(0, 5))
        
        self.open_folder_button = ttk.Button(results_frame, text="Open Result Folder", command=self.open_result_folder, state=tk.DISABLED)
        self.open_folder_button.pack(fill=tk.X)
        
        # 版權信息
        copyright_label = ttk.Label(left_frame, text="© 2025 Zhen-Tang Huang \n& Chunghwa Telecom Co., Ltd. \n& National Taiwan University Hospital \nAll rights reserved", font=("Arial", 8))
        copyright_label.pack(side=tk.BOTTOM, pady=10)
        
        #---------- 右側面板內容 ----------#
        # 影像瀏覽區域
        self.image_frame = ttk.LabelFrame(right_frame, text="Analysis Result Preview", padding=10)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 創建一個畫布容器以支持縮放和平移
        self.canvas = tk.Canvas(self.image_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 為畫布添加縮放和平移事件
        self.canvas.bind("<MouseWheel>", self.zoom)  # Windows滾輪事件
        self.canvas.bind("<Button-4>", self.zoom)    # Linux向上滾動
        self.canvas.bind("<Button-5>", self.zoom)    # Linux向下滾動
        
        # 添加拖拽功能
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)
        
        # 影像瀏覽控制
        nav_frame = ttk.Frame(self.image_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        # 導航按鈕和計數器
        nav_buttons_frame = ttk.Frame(nav_frame)
        nav_buttons_frame.pack(fill=tk.X)
        
        # 左側按鈕
        left_frame = ttk.Frame(nav_buttons_frame)
        left_frame.pack(side=tk.LEFT, expand=True)
        self.prev_button = ttk.Button(left_frame, text="Previous", command=self.show_prev_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT)
        
        # 中間計數器
        center_frame = ttk.Frame(nav_buttons_frame)
        center_frame.pack(side=tk.LEFT, expand=True)
        self.image_counter = ttk.Label(center_frame, text="0/0")
        self.image_counter.pack(side=tk.TOP)
        
        # 右側按鈕
        right_frame = ttk.Frame(nav_buttons_frame)
        right_frame.pack(side=tk.RIGHT, expand=True)
        self.next_button = ttk.Button(right_frame, text="Next", command=self.show_next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.RIGHT)
        
        # 縮放控制按鈕 - 居中布局
        zoom_frame = ttk.Frame(nav_frame)
        zoom_frame.pack(fill=tk.X, pady=(5, 0))
        
        # 使按鈕居中
        zoom_buttons_container = ttk.Frame(zoom_frame)
        zoom_buttons_container.pack(side=tk.TOP, fill=tk.X)
        
        # 左側填充空間
        ttk.Frame(zoom_buttons_container).pack(side=tk.LEFT, expand=True)
        
        # 縮放按鈕在中間
        zoom_buttons = ttk.Frame(zoom_buttons_container)
        zoom_buttons.pack(side=tk.LEFT)
        
        self.zoom_out_button = ttk.Button(zoom_buttons, text="Zoom Out (-)", command=self.zoom_out, state=tk.DISABLED)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)
        
        self.zoom_reset_button = ttk.Button(zoom_buttons, text="Reset Zoom (100%)", command=self.zoom_reset, state=tk.DISABLED)
        self.zoom_reset_button.pack(side=tk.LEFT, padx=5)
        
        self.zoom_in_button = ttk.Button(zoom_buttons, text="Zoom In (+)", command=self.zoom_in, state=tk.DISABLED)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)
        
        # 右側填充空間
        ttk.Frame(zoom_buttons_container).pack(side=tk.LEFT, expand=True)
    
    def browse_image_dir(self):
        dirname = filedialog.askdirectory(initialdir=self.image_dir.get())
        if dirname:
            self.image_dir.set(dirname)
    
    def browse_json_dir(self):
        dirname = filedialog.askdirectory(initialdir=self.json_dir.get())
        if dirname:
            self.json_dir.set(dirname)
    
    def browse_output_dir(self):
        dirname = filedialog.askdirectory(initialdir=self.output_dir.get())
        if dirname:
            self.output_dir.set(dirname)
    
    def update_progress(self, current, total, message=""):
        progress = int((current + 1) / total * 100)
        self.progress_var.set(progress)
        self.status_var.set(message)
        self.master.update_idletasks()
    
    def start_analysis(self):
        # 檢查路徑是否有效
        if not os.path.exists(self.image_dir.get()):
            messagebox.showerror("Error", "Image directory does not exist!")
            return
        
        if not os.path.exists(self.json_dir.get()):
            messagebox.showerror("Error", "Annotation directory does not exist!")
            return
        
        # 禁用按鈕
        self.analyze_button.configure(state=tk.DISABLED)
        
        # 重置進度
        self.progress_var.set(0)
        self.status_var.set("Starting analysis...")
        
        # 在新線程中運行分析
        self.analysis_thread = threading.Thread(target=self.run_analysis)
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def run_analysis(self):
        try:
            # 獲取轉換比例
            try:
                px_per_mm = float(self.px_to_mm_ratio.get())
                if px_per_mm <= 0:
                    raise ValueError("Conversion ratio must be positive")
            except ValueError:
                # 如果輸入無效，使用預設值
                messagebox.showwarning("Warning", "Invalid conversion ratio. Using default (1mm = 19px).")
                px_per_mm = 19
                self.px_to_mm_ratio.set("19")
            
            # 創建分析器並執行分析
            analyzer = ParathyroidTumorAnalyzer(
                self.image_dir.get(), 
                self.json_dir.get(), 
                self.output_dir.get(),
                px_per_mm,  # 傳遞轉換比例
                self.update_progress
            )
            analyzer.analyze_all_images()
            self.results_csv = os.path.join(analyzer.output_dir, "parathyroid_analysis_results.csv")
            self.processed_images = analyzer.processed_images
            
            # 完成後啟用按鈕
            self.master.after(0, self.analysis_complete)
        except Exception as e:
            # self.master.after(0, lambda: self.show_error(f"Error during analysis: {str(e)}"))
            messagebox.showerror("Error", f"Error during analysis: {str(e)}")
    
    def analysis_complete(self):
        self.status_var.set("Analysis completed")
        self.analyze_button.configure(state=tk.NORMAL)
        self.open_csv_button.configure(state=tk.NORMAL)
        self.open_folder_button.configure(state=tk.NORMAL)
        
        # 如果有結果影像，顯示第一張
        if self.processed_images:
            self.current_image_index = 0
            self.show_image(self.processed_images[0])
            
            # 啟用導航按鈕
            self.prev_button.configure(state=tk.NORMAL)
            self.next_button.configure(state=tk.NORMAL)
            self.zoom_in_button.configure(state=tk.NORMAL)
            self.zoom_out_button.configure(state=tk.NORMAL)
            self.zoom_reset_button.configure(state=tk.NORMAL)
            
            # 更新計數器
            self.image_counter.configure(text=f"1/{len(self.processed_images)}")
        
        messagebox.showinfo("Complete", "Image analysis completed!")
    
    def show_error(self, message):
        self.status_var.set("Error")
        self.analyze_button.configure(state=tk.NORMAL)
        messagebox.showerror("Error", message)
    
    def open_csv(self):
        if self.results_csv and os.path.exists(self.results_csv):
            try:
                # 嘗試使用系統默認程序打開CSV文件
                if os.name == 'nt':  # Windows
                    os.startfile(self.results_csv)
                elif os.name == 'posix':  # macOS, Linux
                    os.system(f"open '{self.results_csv}'")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot open CSV file: {str(e)}")
    
    def open_result_folder(self):
        if os.path.exists(self.output_dir.get()):
            try:
                # 嘗試使用系統默認文件管理器打開結果目錄
                if os.name == 'nt':  # Windows
                    os.startfile(self.output_dir.get())
                elif os.name == 'posix':  # macOS, Linux
                    os.system(f"open '{self.output_dir.get()}'")
            except Exception as e:
                messagebox.showerror("Error", f"Cannot open result directory: {str(e)}")
    
    def show_image(self, image_path):
        try:
            if os.path.exists(image_path):
                # 清除畫布
                self.canvas.delete("all")
                
                # 讀取圖像
                self.original_image = Image.open(image_path)
                self.current_zoom = 1.0
                
                # 重置拖拽狀態
                self.drag_data = {"x": 0, "y": 0, "dragging": False}
                self.can_drag = False  # 初始禁用拖動
                
                # 調整圖像以適應畫布
                self.update_image()
                
                # 更新圖像計數器
                self.image_counter.configure(text=f"{self.current_image_index + 1}/{len(self.processed_images)}")
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display image: {str(e)}")
    
    def update_image(self):
        if self.original_image:
            # 獲取畫布大小
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # 如果畫布尚未配置（第一次加載時）
            if canvas_width <= 1:
                canvas_width = 600
                canvas_height = 400
            
            # 獲取原始圖像尺寸
            img_width, img_height = self.original_image.size
            
            # 清除當前畫布內容
            self.canvas.delete("all")
            self.image_item = None
            
            # 如果是初始顯示（縮放因子為1.0），則自動適應畫布大小
            if self.current_zoom == 1.0:
                # 計算縮放以適應畫布
                scale = min(canvas_width / img_width, canvas_height / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                resized_img = self.original_image.resize((new_width, new_height), Image.LANCZOS)
                self.can_drag = False  # 在1:1比例下禁用拖動
            else:
                # 依據縮放因子調整圖像大小
                new_width = int(img_width * self.current_zoom)
                new_height = int(img_height * self.current_zoom)
                resized_img = self.original_image.resize((new_width, new_height), Image.LANCZOS)
                self.can_drag = True  # 在放大狀態下啟用拖動
            
            # 將PIL圖像轉換為Tkinter的PhotoImage
            self.tk_img = ImageTk.PhotoImage(resized_img)
            
            # 計算居中位置
            x_position = (canvas_width - new_width) // 2
            y_position = (canvas_height - new_height) // 2
            
            # 在畫布上顯示圖像，並保存圖像項目ID
            self.image_item = self.canvas.create_image(x_position, y_position, anchor="nw", image=self.tk_img)
    
    # 拖拽相關方法
    def start_drag(self, event):
        # 只有當放大到一定程度時才能拖拽，避免在1:1比例時拖拽
        if self.can_drag:
            # 記錄鼠標起始位置
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            self.drag_data["dragging"] = True
    
    def drag(self, event):
        # 確保圖片存在且處於拖拽狀態
        if self.drag_data["dragging"] and self.image_item is not None and self.can_drag:
            # 計算移動距離
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            # 移動圖像
            self.canvas.move(self.image_item, dx, dy)
            
            # 更新鼠標位置
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
    
    def stop_drag(self, event):
        # 結束拖拽
        self.drag_data["dragging"] = False
    
    def zoom(self, event):
        if not self.original_image:
            return
            
        # 記錄縮放前的圖片位置（如果存在）
        old_zoom = self.current_zoom
            
        # 處理不同系統上的滾輪事件
        if event.num == 4 or event.delta > 0:  # 放大
            self.current_zoom *= 1.1
        elif event.num == 5 or event.delta < 0:  # 縮小
            self.current_zoom *= 0.9
        
        # 限制縮放範圍
        self.current_zoom = max(0.1, min(self.current_zoom, 5.0))
        
        # 更新圖像
        self.update_image()
    
    def zoom_in(self):
        if self.original_image:
            self.current_zoom *= 1.2
            self.current_zoom = min(self.current_zoom, 5.0)
            self.update_image()
    
    def zoom_out(self):
        if self.original_image:
            self.current_zoom *= 0.8
            self.current_zoom = max(self.current_zoom, 0.1)
            self.update_image()
    
    def zoom_reset(self):
        if self.original_image:
            self.current_zoom = 1.0
            self.drag_data = {"x": 0, "y": 0, "dragging": False}  # 重置拖拽狀態
            self.update_image()
    
    def show_prev_image(self):
        if self.processed_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.processed_images)
            self.show_image(self.processed_images[self.current_image_index])
    
    def show_next_image(self):
        if self.processed_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.processed_images)
            self.show_image(self.processed_images[self.current_image_index])


if __name__ == "__main__":
    # 創建並運行GUI
    root = tk.Tk()
    app = ParathyroidAnalyzerGUI(root)
    root.mainloop()
    #
