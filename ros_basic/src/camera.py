import cv2
import numpy as np

# === 1. Đọc ảnh ===
img = cv2.imread("a.jpg")  # đổi tên file ảnh của bạn
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === 2. Chuyển sang HLS ===
img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)

# === 3. Mask cho đường phân chia ở giữa (màu vàng) ===
# Hue: 15-35, Saturation cao, Lightness trung bình
lower_yellow = np.array([15, 30, 115])
upper_yellow = np.array([35, 204, 255])
mask_yellow = cv2.inRange(img_hls, lower_yellow, upper_yellow)

# === 4. Mask cho 2 lane ngoài (màu trắng hoặc xám sáng) ===
# White lane có L cao, S thấp (ít bão hòa)
lower_white = np.array([0, 200, 0])
upper_white = np.array([180, 255, 255])
mask_white = cv2.inRange(img_hls, lower_white, upper_white)

# Xám sáng cũng có thể thuộc vùng này, nên thêm chút dung sai
lower_gray = np.array([0, 150, 0])
upper_gray = np.array([180, 220, 120])
mask_gray = cv2.inRange(img_hls, lower_gray, upper_gray)

# Gộp trắng + xám
mask_outer = cv2.bitwise_or(mask_white, mask_gray)

# === 5. Hiển thị ảnh kết quả ===
# Gộp lại các ảnh cho trực quan
img_hls_vis = cv2.cvtColor(img_hls, cv2.COLOR_HLS2RGB)

cv2.imshow("1. Anh goc (RGB)", img)
cv2.imshow("2. Anh HLS", img_hls_vis)
cv2.imshow("3. Mask duong giua (mau vang)", mask_yellow)
cv2.imshow("4. Mask lane ngoai (trang/xam)", mask_outer)

cv2.waitKey(0)
cv2.destroyAllWindows()