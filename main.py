import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove
import math
from scipy.spatial import KDTree
import time
import os
import uuid
from flask import Flask, request, render_template, send_from_directory, jsonify
import json

class ImageProcessor:
    def __init__(self):
        self.points = []
        
    def order_points(self, pts):
        """按順序排列四個點"""
        rect = np.zeros((4, 2), dtype="float32")
        s = np.sum(pts, axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上角
        rect[2] = pts[np.argmax(s)]  # 右下角

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上角
        rect[3] = pts[np.argmax(diff)]  # 左下角

        return rect

    def four_point_transform(self, image, pts):
        """透視變換函數"""
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # 計算新的寬度和高度
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = max(int(width_top), int(width_bottom))

        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = max(int(height_left), int(height_right))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # 計算透視變換矩陣並應用
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))

        return warped

    def on_click(self, event):
        """用於處理matplotlib的點擊事件"""
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.points.append((x, y))
            orig_copy = self.orig.copy()
            cv2.circle(orig_copy, (x, y), 5, (0, 0, 255), -1)
            plt.imshow(cv2.cvtColor(orig_copy, cv2.COLOR_BGR2RGB))
            plt.title("Click to select points")
            plt.axis('off')
            plt.draw()

            if len(self.points) == 4:
                print("4個點已標記完成！")
                plt.close()

    def A4_transform(self, image_path, points):
        """
        根據前端傳來的四個點資料進行透視變換

        :param image_path: 圖片路徑
        :param points: 四個點的列表，每個點可以是 tuple 或 dict 格式
        :return: 透視轉換後的圖片儲存路徑
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"無法讀取圖片 {image_path}")

        # 如果點資料為 dict 格式，轉換成 tuple
        if isinstance(points[0], dict):
            points = [(p['x'], p['y']) for p in points]
    
        points_array = np.array(points, dtype="float32")
        if points_array.shape != (4, 2):
            raise ValueError("請提供四個點，每個點需包含 x 與 y 座標")
    
        warped_image = self.four_point_transform(image, points_array)
    
        output_dir = 'uploads'
        os.makedirs(output_dir, exist_ok=True)
        original_filename = os.path.basename(image_path)
        filename_without_ext = os.path.splitext(original_filename)[0]
        output_filename = f"{filename_without_ext}_transformed.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, warped_image)
        print(f"轉換後的圖片已儲存至: {output_path}")
        return output_path


  
def process_insole(image_path):
    """
    Process insole image: remove background, rotate to correct orientation,
    and measure key dimensions.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        output_filename: Filename of the processed image
        result_data: Dictionary containing measurement results
    """
    start_time = time.time()
    
    # 讀取圖像
    input_image = cv2.imread(image_path)
    
    # 檢查圖片是否成功讀取
    if input_image is None:
        raise Exception("Failed to read the image file")
    
    # 調整圖片大小
    max_dimension = 2000
    height, width = input_image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        input_image = cv2.resize(input_image, None, fx=scale, fy=scale)
    
    ##### 一、前處理 #####
    # 使用 rembg 進行去背
    segmented_image = remove(input_image)
    
    # 將去背後的圖像轉換為 3 通道（BGR），以便後續處理
    if segmented_image.shape[2] == 4:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGRA2BGR)
    
    # 將圖片轉成灰階
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    
    # 增強對比度
    gray = cv2.convertScaleAbs(gray, alpha=5, beta=0)
    
    # 高斯模糊，去除噪點
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 邊緣檢測 - 使用Canny邊緣檢測
    edges = cv2.Canny(blurred, 50, 150)
    
    # 閉運算處理，填補輪廓內的小空隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 找到邊緣的輪廓
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise Exception("No contours found in the image")
    
    # 找到最大的輪廓，假設它是鞋墊的輪廓
    max_contour = max(contours, key=cv2.contourArea)
    
    ##### 二、抓特徵點與旋轉 #####
    # 使用 KD 樹加速查找最遠距離的兩點
    points = np.array([point[0] for point in max_contour])  # 提取輪廓上所有點
    kdtree = KDTree(points)  # 創建 KD 樹
    
    # 設定變數:最大距離、對應的兩個點
    max_distance = 0
    topmost, bottommost = None, None
    
    # 遍歷輪廓中任意兩點，使用KD樹求出距離最長的兩個點
    for i, point in enumerate(points):
        dist, idx = kdtree.query(point, k=points.shape[0])
        farthest_point = points[idx[-1]]  # 取最遠的那個點
        distance = np.linalg.norm(point - farthest_point)
        
        # 更新最大距離和兩點
        if distance > max_distance:
            max_distance = distance
            topmost, bottommost = point, farthest_point
    
    # 旋轉圖像的輔助函數
    def calculate_rotation_angle(topmost, bottommost):
        vector_line = (bottommost[0] - topmost[0], bottommost[1] - topmost[1])
        vector_mid_line = (0, 1)  
        dot_product = vector_line[0] * vector_mid_line[0] + vector_line[1] * vector_mid_line[1]
        magnitude_line = math.sqrt(vector_line[0] ** 2 + vector_line[1] ** 2)
        magnitude_mid_line = math.sqrt(vector_mid_line[0] ** 2 + vector_mid_line[1] ** 2)
        
        cos_angle = dot_product / (magnitude_line * magnitude_mid_line)
        cos_angle = max(-1, min(1, cos_angle))
        angle_radians = math.acos(cos_angle)
        return math.degrees(angle_radians)
    
    def rotate_image(image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    # 根據 topmost 和 bottommost 的位置判斷左斜或右斜
    if topmost[1] < bottommost[1]:
        print("左斜")   
        # 計算旋轉角度
        angle_degrees = calculate_rotation_angle(topmost, bottommost)
        # 旋轉圖像
        rotated_image = rotate_image(segmented_image, -angle_degrees)
    elif topmost[1] > bottommost[1]:
        print("右斜")
        # 計算旋轉角度
        angle_degrees = 180 - calculate_rotation_angle(topmost, bottommost)
        # 旋轉圖像
        rotated_image = rotate_image(segmented_image, angle_degrees)
    else:
        # 若上下點垂直對齊，不需要旋轉
        rotated_image = segmented_image
    
    ##### 三、重新獲取旋轉後的輪廓 #####
    # 轉灰階、強化對比
    rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    rotated_gray = cv2.convertScaleAbs(rotated_gray, alpha=6, beta=0)
    rotated_blurred = cv2.GaussianBlur(rotated_gray, (7, 7), 0)
    rotated_edges = cv2.Canny(rotated_blurred, 150, 220)
    
    # 找輪廓
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    rotated_closed_edges = cv2.morphologyEx(rotated_edges, cv2.MORPH_CLOSE, kernel)
    rotated_contours, _ = cv2.findContours(rotated_closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not rotated_contours:
        raise Exception("No contours found in the rotated image")
    
    rotated_max_contour = max(rotated_contours, key=cv2.contourArea)
    
    ##### 四、獲取關鍵點並計算尺寸 #####
    # 獲取鞋墊的最頂點和最底點（垂直方向）
    bottom_point = tuple(rotated_max_contour[rotated_max_contour[:, :, 1].argmax()][0])
    front_point = tuple(rotated_max_contour[rotated_max_contour[:, :, 1].argmin()][0])
    foot_length_pixels = math.dist(bottom_point, front_point)
    
    # 計算 A4 紙比例換算長度（像素對實際長度）
    a4_width_cm = 21.0
    a4_length_cm = 29.7
    image_height, image_width = rotated_image.shape[:2]
    pixels_per_cm = ((image_width / a4_width_cm) + (image_height / a4_length_cm)) / 2
    
    # 計算鞋墊長度
    insole_length_cm = foot_length_pixels / pixels_per_cm
    
    def draw_line_and_length(image, point1, point2, length_cm, color, label, line_thickness=10):
        cv2.line(image, point1, point2, color, line_thickness)
        mid_point = (int((point1[0] + point2[0]) / 2), int((point1[1] + point2[1]) / 2))
        cv2.putText(image, f"{label}: {length_cm:.2f} cm", mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # 繪製足長（藍線）
    draw_line_and_length(rotated_image, bottom_point, front_point, insole_length_cm, (255, 135, 0), "Length")
    cv2.circle(rotated_image, bottom_point, 12, (255, 135, 0), -1)  # 藍色圓點
    cv2.circle(rotated_image, front_point, 12, (255, 135, 0), -1)  # 藍色圓點
    
    # 前掌寬計算（紅線）
    y_threshold = front_point[1] + int(foot_length_pixels * 0.5)
    front_half_points = [pt[0] for pt in rotated_max_contour if pt[0][1] <= y_threshold]
    
    forefoot_width = 0
    if front_half_points:
        left_most = tuple(min(front_half_points, key=lambda x: x[0]))
        right_most = tuple(max(front_half_points, key=lambda x: x[0]))
        forefoot_width = math.dist(left_most, right_most) / pixels_per_cm
        draw_line_and_length(rotated_image, left_most, right_most, forefoot_width, (0, 0, 255), "Forefoot")
        forefoot_center = ((left_most[0] + right_most[0]) // 2, (left_most[1] + right_most[1]) // 2)
        cv2.circle(rotated_image, forefoot_center, 12, (0, 0, 255), -1)  # 前掌中心點
    
    # 中足寬計算的輔助函數
    def find_nearest_contour_point(image, start_point, direction, contour, target_y):
        x, y = start_point
        step = 5 if direction == "right" else -5
        closest_point = None
        min_dist = float('inf')
        
        for offset_y in range(-20, 20, 1):  
            x_temp, y_temp = x, y + offset_y
            while 0 <= x_temp < image.shape[1]:
                point = (float(x_temp), float(y_temp))
                dist = cv2.pointPolygonTest(contour, point, True)
                if dist >= 0 and abs(dist) < min_dist:
                    min_dist = abs(dist)
                    closest_point = (int(x_temp), target_y)  
                x_temp += step
        return closest_point
    
    # 中足寬計算（紫線）
    midfoot_y = int(bottom_point[1] - 0.4 * foot_length_pixels)
    midfoot_pt = (bottom_point[0], midfoot_y)
    
    left_point = find_nearest_contour_point(rotated_image, midfoot_pt, "left", rotated_max_contour, midfoot_pt[1])
    right_point = find_nearest_contour_point(rotated_image, midfoot_pt, "right", rotated_max_contour, midfoot_pt[1])
    midfoot_width = 0
    
    if left_point and right_point:
        midfoot_width = math.dist(left_point, right_point) / pixels_per_cm
        draw_line_and_length(rotated_image, left_point, right_point, midfoot_width, (255, 0, 155), "Midfoot")
        cv2.circle(rotated_image, left_point, 12, (255, 0, 155), -1)  # 紫色圓點
        cv2.circle(rotated_image, right_point, 12, (255, 0, 155), -1)  # 紫色圓點
    
    # 後跟寬計算（黃線）
    heel_offset = int(0.15 * foot_length_pixels)
    heel_y = bottom_point[1] - heel_offset
    heel_center = (bottom_point[0], heel_y)
    
    heel_width_left = find_nearest_contour_point(rotated_image, heel_center, "left", rotated_max_contour, heel_y)
    heel_width_right = find_nearest_contour_point(rotated_image, heel_center, "right", rotated_max_contour, heel_y)
    
    heel_width = 0

    if heel_width_left and heel_width_right and forefoot_center:
        heel_width = math.dist(heel_width_left, heel_width_right) / pixels_per_cm
        draw_line_and_length(rotated_image, heel_width_left, heel_width_right, heel_width, (0, 255, 255), "Heel")
        cv2.circle(rotated_image, heel_width_left, 12, (0, 255, 255), -1)  # 黃色圓點
        cv2.circle(rotated_image, heel_width_right, 12, (0, 255, 255), -1)  # 黃色圓點
        # 計算後跟中心點
        heel_center_point = ((heel_width_left[0] + heel_width_right[0]) // 2, heel_y)
    
        # 從後跟中心點直接畫到前掌中心點,然後延長至碰到鞋墊邊緣
        green_line_start = heel_center_point
        green_line_end = forefoot_center
        line_direction = np.array([green_line_end[0] - green_line_start[0], 
                                 green_line_end[1] - green_line_start[1]], dtype=float)
        line_length = np.linalg.norm(line_direction)
        line_direction /= line_length
        step_size = 15
        green_line_endpoint = None
    
        # 延長線直到碰到輪廓
        current_end = green_line_end
        while True:
            current_end = (int(current_end[0] + line_direction[0] * step_size), 
                          int(current_end[1] + line_direction[1] * step_size))
            if cv2.pointPolygonTest(rotated_max_contour, current_end, True) < 0:
                green_line_endpoint = current_end
                break
    
        if green_line_endpoint is not None:
            cv2.circle(rotated_image, green_line_endpoint, 12, (0, 255, 0), -1)  # 綠色圓點
            cv2.circle(rotated_image, green_line_start, 12, (0, 255, 255), -1)  # 黃色圓點
            cv2.line(rotated_image, green_line_start, green_line_endpoint, (0, 255, 0), 10)  # 使用固定線寬10
    else:
        print("未找到必要的點以繪製中心線。")   
    # 生成唯一的輸出檔名
    output_filename = f"result_{uuid.uuid4().hex[:8]}.png"
    output_folder = "static"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, rotated_image)
    
    elapsed_time = time.time() - start_time
    result_data = {
        "length_cm": round(insole_length_cm, 2),
        "forefoot_width_cm": round(forefoot_width, 2),
        "midfoot_width_cm": round(midfoot_width, 2),
        "heel_width_cm": round(heel_width, 2),
        "processing_time": round(elapsed_time, 2)
    }
    
    return output_filename, result_data

# Flask 應用部分
from flask import Flask, request, render_template, send_from_directory, jsonify

app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        
        if file.filename == 'camera-capture.jpg':
            unique_filename = f"camera_{uuid.uuid4().hex[:8]}.jpg"
        else:
            unique_filename = file.filename
            
        filename = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(filename)
        
        try:
            points_data = request.form.get("points")
            if points_data:
                # ① 根據前端傳來的四個點進行透視轉換
                points = json.loads(points_data)
                processor = ImageProcessor()
                warped_path = processor.A4_transform(filename, points)
                
                # ② 對平面化後的圖片，繼續進行背景去背、旋轉校正、輪廓分析、四條線繪製與尺寸計算
                output_filename, result_data = process_insole(warped_path)
            else:
                # 沒有點資料則使用原本的完整流程處理原圖
                output_filename, result_data = process_insole(filename)
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': True,
                    'image_url': output_filename,
                    'result_data': result_data
                })
                
            return render_template("index.html", image_url=output_filename, result_data=result_data)
            
        except Exception as e:
            error_message = str(e)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': False,
                    'error': error_message
                }), 400
            return f"Error processing image: {error_message}", 400
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    return render_template("index.html", image_url=None, result_data=None)
    return render_template("index.html", image_url=None, result_data=None)


@app.route("/static/<filename>")
def get_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# 單獨測試處理函數
if __name__ == "__main__":
    # 啟動 Flask 應用
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)