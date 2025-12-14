"""
Script nhận dạng biển số xe từ ảnh tĩnh.

Quy trình chính:
1. Tải mô hình KNN đã huấn luyện.
2. Tiền xử lý ảnh: grayscale, threshold, Canny edge detection.
3. Phát hiện contours và lọc ra các contour có dạng hình chữ nhật (biển số).
4. Với mỗi biển số phát hiện: xoay thẳng, cắt ROI, phân đoạn ký tự.
5. Nhận dạng từng ký tự bằng KNN và ghép thành chuỗi biển số.

Kết quả: Hiển thị biển số nhận dạng được trên ảnh và in ra console.
"""

import math

import cv2
import numpy as np

import Preprocess

# Các hằng số cho ngưỡng hóa thích ứng
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

# Biến đếm biển số
n = 1

# Ngưỡng diện tích ký tự (tỷ lệ so với diện tích biển số)
Min_char = 0.01
Max_char = 0.09

# Kích thước resize ký tự cho KNN (20x30 pixels)
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Đọc ảnh đầu vào và resize về 1920x1080 để xử lý
img = cv2.imread("data/image/1.jpg")
img = cv2.resize(img, dsize=(1920, 1080))



# Tải mô hình KNN đã huấn luyện từ file
# classifications.txt: nhãn của các ký tự (mã ASCII)
# flattened_images.txt: dữ liệu ảnh phẳng của ký tự mẫu
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))
# Tạo và huấn luyện mô hình KNN
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

# Tiền xử lý ảnh: chuyển xám, ngưỡng hóa
imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
# Phát hiện cạnh bằng Canny để tìm đường viền biển số
canny_image = cv2.Canny(imgThreshplate, 250, 255)
# Kernel cho dilation
kernel = np.ones((3, 3), np.uint8)
# Mở rộng cạnh để kết nối các đường viền
dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

# Tìm contours từ ảnh dilated
contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sắp xếp và lấy 10 contours có diện tích lớn nhất (có thể là biển số)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Danh sách lưu các contour có dạng hình chữ nhật (biển số tiềm năng)
screenCnt = []
for c in contours:
    # Tính chu vi của contour
    peri = cv2.arcLength(c, True)
    # Xấp xỉ contour thành đa giác với độ chính xác 6% chu vi
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)
    # Lấy bounding box
    [x, y, w, h] = cv2.boundingRect(approx.copy())
    ratio = w / h
    # Chỉ giữ lại contour có 4 đỉnh (tứ giác) - dạng biển số
    if (len(approx) == 4):
        screenCnt.append(approx)
        # Hiển thị số đỉnh trên ảnh (debug)
        cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

# Kiểm tra có phát hiện biển số không
if screenCnt is None:
    detected = 0
    print("No plate detected")
else:
    detected = 1

if detected == 1:
    # Duyệt qua từng biển số phát hiện
    for screenCnt in screenCnt:
        # Vẽ contour xanh lá để khoanh vùng biển số trên ảnh gốc
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

        # Tính góc nghiêng của biển số dựa trên vị trí 4 góc
        # Giả sử biển số là hình chữ nhật nghiêng
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # Sắp xếp theo y giảm dần để tìm 2 điểm trên cùng
        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]  # Điểm trên trái
        (x2, y2) = array[1]  # Điểm trên phải
        doi = abs(y1 - y2)  # Chiều cao
        ke = abs(x1 - x2)   # Chiều rộng
        # Tính góc nghiêng (đơn vị độ)
        angle = math.atan(doi / ke) * (180.0 / math.pi)

        # Tạo mặt nạ để cắt biển số từ ảnh
        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)

        # Tìm tọa độ min/max của mask để cắt ROI
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        # Cắt biển số từ ảnh gốc và ảnh ngưỡng
        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        # Tính tâm của biển số để xoay
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

        # Xoay biển số để thẳng dựa trên hướng nghiêng
        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

        # Áp dụng phép biến đổi affine để xoay
        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        # Phóng to biển số 3 lần để dễ nhận dạng ký tự
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

        # Tiền xử lý và phân đoạn ký tự trên biển số
        # Tạo kernel hình chữ nhật 3x3 cho morphology
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Dilate để kết nối các phần của ký tự
        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
        # Tìm contours của các ký tự
        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Hiển thị ảnh morphology (debug)
        cv2.imshow(str(n + 20), thre_mor)
        # Vẽ contours vàng của ký tự trên biển số
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)

        # Lọc ra các ký tự hợp lệ dựa trên diện tích và tỷ lệ khung hình
        char_x_ind = {}  # Dictionary lưu vị trí x -> index contour
        char_x = []      # Danh sách vị trí x của ký tự hợp lệ
        height, width, _ = roi.shape
        roiarea = height * width  # Diện tích biển số

        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h  # Tỷ lệ khung hình (width/height)
            char_area = w * h  # Diện tích ký tự

            # Điều kiện lọc: diện tích trong khoảng và tỷ lệ khung hình hợp lý
            if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                if x in char_x:  # Nếu trùng vị trí x, tăng x lên 1 để tránh trùng
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind

        # Nhận dạng ký tự sử dụng mô hình KNN
        char_x = sorted(char_x)  # Sắp xếp theo vị trí x từ trái sang phải
        strFinalString = ""
        first_line = ""   # Dòng trên (thường là chữ cái)
        second_line = ""  # Dòng dưới (thường là số)

        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            # Vẽ hình chữ nhật xanh quanh ký tự
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Cắt ROI của ký tự từ ảnh morphology
            imgROI = thre_mor[y:y + h, x:x + w]

            # Resize về kích thước chuẩn 20x30 cho KNN
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

            npaROIResized = np.float32(npaROIResized)
            # Dự đoán ký tự bằng KNN với k=3
            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)
            # Chuyển mã ASCII thành ký tự
            strCurrentChar = str(chr(int(npaResults[0][0])))
            # Hiển thị ký tự dự đoán trên biển số (màu xanh dương)
            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

            # Phân biệt biển số 1 dòng hay 2 dòng dựa trên vị trí y
            # Nếu y < 1/3 chiều cao -> dòng trên, ngược lại dòng dưới
            if (y < height / 3):
                first_line = first_line + strCurrentChar
            else:
                second_line = second_line + strCurrentChar

        # In kết quả biển số ra console
        print("\n Biển số xe " + str(n) + " là: " + first_line + " - " + second_line + "\n")
        # Resize biển số về 75% để hiển thị
        roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        # Hiển thị biển số với ký tự nhận dạng
        cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        n = n + 1

img = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imshow('License plate', img)

cv2.waitKey(0)
