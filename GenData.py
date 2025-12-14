# GenData.py

import numpy as np
import cv2
import sys


# module level variables ##########################################################################
MIN_CONTOUR_AREA = 40


RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

###################################################################################################
def main():
    """
    Hàm chính để tạo dữ liệu huấn luyện cho nhận dạng ký tự biển số xe.

    Quy trình:
    1. Đọc ảnh mẫu training_chars.png chứa các ký tự mẫu.
    2. Xử lý ảnh: chuyển xám, làm mờ, ngưỡng hóa để tách ký tự.
    3. Tìm contours (đường viền) của các ký tự.
    4. Với mỗi contour đủ lớn, cắt ROI, resize và hiển thị để người dùng gán nhãn.
    5. Lưu nhãn và dữ liệu ảnh phẳng vào file.

    Kết quả:
    - classifications.txt: Chứa nhãn của các ký tự (mã ASCII).
    - flattened_images.txt: Chứa dữ liệu ảnh phẳng (20x30 = 600 pixel mỗi ảnh).
    """
    # Đọc ảnh mẫu chứa các ký tự huấn luyện
    imgTrainingNumbers = cv2.imread("training_chars.png")
    # Có thể resize nếu cần, nhưng hiện tại comment out
    #imgTrainingNumbers = cv2.resize(imgTrainingNumbers, dsize = None, fx = 0.5, fy = 0.5)

    # Chuyển ảnh sang xám để xử lý đơn giản hơn
    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)
    # Làm mờ bằng Gauss để giảm nhiễu
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)

    # Ngưỡng hóa thích ứng để chuyển thành ảnh nhị phân (chữ trắng, nền đen)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                       255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       11,
                                       2)

    # Hiển thị ảnh ngưỡng để kiểm tra
    cv2.imshow("imgThresh", imgThresh)

    # Sao chép ảnh ngưỡng để tìm contours
    imgThreshCopy = imgThresh.copy()

    # Tìm contours (đường viền) của các ký tự
    npaContours, hierarchy = cv2.findContours(imgThreshCopy,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

    # Khởi tạo mảng để lưu ảnh phẳng (flattened images)
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    # Danh sách lưu nhãn (classifications) của các ký tự
    intClassifications = []

    # Danh sách các ký tự hợp lệ bao gồm chữ số từ 0 đến 9 và chữ cái từ A đến Z, được biểu diễn bằng mã ASCII
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                      ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                      ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                      ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    # Duyệt qua từng contour
    for npaContour in npaContours:
        # Chỉ xử lý contour có diện tích > MIN_CONTOUR_AREA để loại bỏ nhiễu
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:
            # Lấy bounding box của contour
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)

            # Vẽ hình chữ nhật đỏ quanh ký tự trên ảnh gốc
            cv2.rectangle(imgTrainingNumbers,
                           (intX, intY),
                           (intX+intW,intY+intH),
                           (0, 0, 255),
                           2)

            # Cắt vùng quan tâm (ROI) từ ảnh ngưỡng
            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]
            # Resize ROI về kích thước chuẩn 20x30
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

            # Hiển thị ROI gốc và resized
            cv2.imshow("imgROI", imgROI)
            cv2.imshow("imgROIResized", imgROIResized)

            # Hiển thị ảnh gốc với bounding box
            cv2.imshow("training_numbers.png", imgTrainingNumbers)

            # Chờ người dùng nhấn phím để gán nhãn
            intChar = cv2.waitKey(0)

            # Nếu nhấn ESC, thoát chương trình
            if intChar == 27:
                sys.exit()
            # Nếu phím nhấn là ký tự hợp lệ, lưu nhãn và dữ liệu ảnh
            elif intChar in intValidChars:
                # Thêm nhãn vào danh sách
                intClassifications.append(intChar)
                # Chuyển ảnh resized thành vector phẳng (1x600)
                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                # Thêm vào mảng tổng
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

            # end if
        # end if
    # end for

    # Chuyển danh sách nhãn thành numpy array kiểu float32
    fltClassifications = np.array(intClassifications, np.float32)
    # Reshape thành cột (n, 1)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

    # In thông báo hoàn thành
    print ("\n\nHoàn thành huấn luyện !!\n")

    # Lưu nhãn vào file classifications.txt
    np.savetxt("classifications.txt", npaClassifications)
    # Lưu dữ liệu ảnh phẳng vào flattened_images.txt
    np.savetxt("flattened_images.txt", npaFlattenedImages)

    # Đóng tất cả cửa sổ OpenCV
    cv2.destroyAllWindows()

    return

###################################################################################################
if __name__ == "__main__":
    main()
