"""
Script nhận dạng biển số xe từ video.

Quy trình tương tự như xử lý ảnh tĩnh nhưng áp dụng cho từng frame của video:
1. Tải mô hình KNN đã huấn luyện.
2. Đọc video frame by frame.
3. Với mỗi frame: tiền xử lý, phát hiện contours, lọc biển số, nhận dạng ký tự.
4. Hiển thị biển số nhận dạng trên video và thống kê tỷ lệ phát hiện.

Kết quả: Video với biển số được khoanh vùng và hiển thị chuỗi ký tự.
"""

import math

import cv2
import numpy as np

import Preprocess

# Các hằng số cho ngưỡng hóa thích ứng
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

# Ngưỡng diện tích ký tự (không dùng trong code hiện tại)
Min_char_area = 0.015
Max_char_area = 0.06

# Ngưỡng diện tích ký tự (tỷ lệ so với biển số)
Min_char = 0.01
Max_char = 0.09

# Ngưỡng tỷ lệ khung hình ký tự (width/height)
Min_ratio_char = 0.25
Max_ratio_char = 0.7

# Ngưỡng kích thước biển số (pixel)
max_size_plate = 18000
min_size_plate = 5000

# Kích thước resize ký tự cho KNN
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

# Biến thống kê
tongframe = 0      # Tổng số frame xử lý
biensotimthay = 0  # Số biển số tìm thấy

# Tải mô hình KNN đã huấn luyện
npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape(
    (npaClassifications.size, 1))
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

# Đọc video từ file
cap = cv2.VideoCapture('data/video/video1.mp4')
# Vòng lặp xử lý từng frame
while (cap.isOpened()):

    ret, img = cap.read()
    tongframe = tongframe + 1
    # Tiền xử lý frame để chuẩn bị phát hiện biển số
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
    # Phát hiện cạnh bằng Canny để tìm đường viền biển số
    canny_image = cv2.Canny(imgThreshplate, 250, 255)
    kernel = np.ones((3, 3), np.uint8)
    # Dilate để làm dày và kết nối các đường viền
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

    # Tìm contours từ ảnh dilated
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sắp xếp và lấy 10 contours có diện tích lớn nhất (tiềm năng là biển số)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tính chu vi
        # Xấp xỉ thành đa giác với độ chính xác 6%
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        # Lọc contour có 4 đỉnh và tỷ lệ khung hình phù hợp với biển số
        # (vuông: 0.8-1.5 hoặc chữ nhật dọc: 4.5-6.5)
        if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
            screenCnt.append(approx)
    # Kiểm tra có phát hiện biển số không
    if screenCnt is None:
        detected = 0
        print("No plate detected")
    else:
        detected = 1

    if detected == 1:
        n = 1
        for screenCnt in screenCnt:
            # Tính góc nghiêng dựa trên 4 đỉnh của biển số
            (x1, y1) = screenCnt[0, 0]
            (x2, y2) = screenCnt[1, 0]
            (x3, y3) = screenCnt[2, 0]
            (x4, y4) = screenCnt[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            sorted_array = array.sort(reverse=True, key=lambda x: x[1])
            (x1, y1) = array[0]  # Điểm trên trái
            (x2, y2) = array[1]  # Điểm trên phải
            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = math.atan(doi / ke) * (180.0 / math.pi)

            # Tạo mask để cắt biển số
            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)

            # Cắt ROI biển số
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = img[topx:bottomx + 1, topy:bottomy + 1]
            imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

            # Xoay biển số thẳng đứng
            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

            # Phóng to 3 lần để dễ nhận dạng
            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

            # Morphology để chuẩn bị phân đoạn ký tự
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
            cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Phân đoạn ký tự trên biển số
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width
            for ind, cnt in enumerate(cont):
                area = cv2.contourArea(cnt)
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                # Lọc ký tự hợp lệ theo diện tích và tỷ lệ
                if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                    if x in char_x:
                        x = x + 1  # Tránh trùng vị trí
                    char_x.append(x)
                    char_x_ind[x] = ind

            # Nhận dạng nếu có 7-9 ký tự (đủ cho biển số Việt Nam)
            if len(char_x) in range(7, 10):
                # Vẽ contour biển số trên frame gốc
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

                char_x = sorted(char_x)  # Sắp xếp từ trái sang phải
                strFinalString = ""
                first_line = ""
                second_line = ""

                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    imgROI = thre_mor[y:y + h, x:x + w]  # Cắt ký tự

                    # Resize về 20x30 cho KNN
                    imgROIResized = cv2.resize(imgROI,
                                                (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                    npaROIResized = imgROIResized.reshape(
                        (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                    npaROIResized = np.float32(npaROIResized)
                    # Dự đoán bằng KNN
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)
                    strCurrentChar = str(chr(int(npaResults[0][0])))
                    # Hiển thị ký tự trên biển số
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 3)

                    # Phân dòng dựa trên vị trí y
                    if (y < height / 3):
                        first_line = first_line + strCurrentChar
                    else:
                        second_line = second_line + strCurrentChar

                strFinalString = first_line + second_line
                print("\n Biển số xe " + str(n) + " là: " + first_line + " - " + second_line + "\n")
                # Hiển thị biển số trên frame
                cv2.putText(img, strFinalString, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                n = n + 1
                biensotimthay = biensotimthay + 1

    # Resize frame để hiển thị
    imgcopy = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow('License plate', imgcopy)
    # In thống kê
    print("number of plates found", biensotimthay)
    print("total frame", tongframe)
    print("plate found rate:", 100 * biensotimthay / (368), "%")

    # Thoát nếu nhấn 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng video và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
