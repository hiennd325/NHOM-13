# Preprocess.py

import cv2
import numpy as np
import math

# Các biến toàn cục ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) # Kích thước bộ lọc Gauss, kích thước càng lớn thì ảnh càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9  

###################################################################################################
def preprocess(imgOriginal):
    """
    Hàm tiền xử lý ảnh để chuẩn bị cho việc nhận dạng biển số xe.

    Tham số:
    imgOriginal: Ảnh gốc đầu vào (màu BGR).

    Trả về:
    imgGrayscale: Ảnh xám gốc.
    imgThresh: Ảnh nhị phân sau khi xử lý.

    Các bước thực hiện:
    1. Chuyển ảnh gốc sang ảnh xám bằng cách trích xuất giá trị cường độ sáng từ kênh V trong HSV.
    2. Tăng độ tương phản để làm nổi bật biển số, giúp dễ dàng tách biển số khỏi nền.
    3. Làm mịn ảnh bằng bộ lọc Gauss để giảm nhiễu, chuẩn bị cho việc ngưỡng hóa.
    4. Tạo ảnh nhị phân bằng ngưỡng hóa thích ứng, chuyển nền sáng thành đen và chữ thành trắng.
    """
    # Chuyển ảnh gốc sang ảnh xám bằng cách trích xuất giá trị cường độ sáng từ kênh V trong HSV
    imgGrayscale = extractValue(imgOriginal)
    # Tăng độ tương phản để làm nổi bật biển số, giúp dễ dàng tách biển số khỏi nền
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    # Làm mịn ảnh bằng bộ lọc Gauss để giảm nhiễu, chuẩn bị cho việc ngưỡng hóa
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

    # Tạo ảnh nhị phân bằng ngưỡng hóa thích ứng, chuyển nền sáng thành đen và chữ thành trắng
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

    # Trả về ảnh xám gốc và ảnh nhị phân đã xử lý
    return imgGrayscale, imgThresh

###################################################################################################
def extractValue(imgOriginal):
    """
    Hàm trích xuất kênh Value từ ảnh HSV để chuyển thành ảnh xám.

    Tham số:
    imgOriginal: Ảnh gốc đầu vào (màu BGR).

    Trả về:
    imgValue: Kênh Value của ảnh HSV, đại diện cho độ sáng.

    Lý do sử dụng HSV:
    - HSV gồm: Hue (màu sắc), Saturation (độ bão hòa), Value (giá trị cường độ sáng).
    - Sử dụng Value vì nó đại diện cho độ sáng, không phụ thuộc vào màu sắc cụ thể như RGB,
      giúp xử lý tốt hơn trong các điều kiện ánh sáng khác nhau.
    """
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    # Chuyển ảnh từ BGR sang HSV để dễ dàng tách các thành phần màu
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    # HSV gồm: Hue (màu sắc), Saturation (độ bão hòa), Value (giá trị cường độ sáng)
    # Sử dụng Value vì nó đại diện cho độ sáng, không phụ thuộc vào màu sắc cụ thể như RGB
    return imgValue

###################################################################################################
def maximizeContrast(imgGrayscale):
    """
    Hàm tăng độ tương phản của ảnh xám bằng phép toán hình thái học.

    Tham số:
    imgGrayscale: Ảnh xám đầu vào.

    Trả về:
    imgGrayscalePlusTopHatMinusBlackHat: Ảnh xám với độ tương phản được tối ưu hóa.

    Nguyên lý:
    - Top-Hat: Làm nổi bật các chi tiết sáng trên nền tối (như chữ trên biển số).
    - Black-Hat: Làm nổi bật các chi tiết tối trên nền sáng.
    - Kết hợp: Cộng ảnh gốc với Top-Hat và trừ Black-Hat để tăng độ tương phản tổng thể.
    - Kernel: Hình chữ nhật 3x3, lặp lại 10 lần để hiệu quả tối ưu.
    """
    # Tăng độ tương phản của ảnh xám để biển số nổi bật hơn
    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    # Tạo kernel hình chữ nhật 3x3 cho phép toán hình thái học
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Áp dụng Top-Hat để làm nổi bật các chi tiết sáng trên nền tối (như chữ trên biển số)
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10)
    # Áp dụng Black-Hat để làm nổi bật các chi tiết tối trên nền sáng
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10)
    # Kết hợp ảnh gốc với Top-Hat và trừ Black-Hat để tăng độ tương phản tổng thể
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # Kết quả là ảnh xám với độ tương phản được tối ưu hóa
    return imgGrayscalePlusTopHatMinusBlackHat










