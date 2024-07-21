import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import threading
import time

# Khởi tạo ORB với số lượng keypoints giới hạn
orb = cv2.ORB_create(nfeatures=10000)  # Giới hạn số lượng keypoints

# Hàm trích xuất đặc trưng ORB
def compute_orb_features(image):
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

# Hàm so khớp descriptors
def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2,k=2)
    good = []
    for(m,n) in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])
    return good

# Hàm tải các ảnh mẫu từ thư mục gốc
def load_template_images(root_folder):
    template_images = {}
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(subfolder_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    scale = 0.4
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)  # Giảm kích thước để tăng tốc
                    kp, des = compute_orb_features(img)
                    if des is not None:
                        template_images[(subfolder, filename)] = (kp, des)
    return template_images

# Hàm so khớp ảnh đầu vào với các mẫu
def match_currency(input_image, template_images):
    kp2, des2 = compute_orb_features(input_image)
    if des2 is None:
        raise ValueError("Không thể trích xuất đặc trưng từ ảnh đầu vào.")

    best_match = None
    best_match_score = 0
    for (subfolder, filename), (kp1, des1) in template_images.items():
        matches = match_features(des1, des2)
        score = len(matches)
        if best_match is None or score > best_match_score:
            best_match = (subfolder, filename)
            best_match_score = score

    return best_match

# Hàm quay video từ camera và nhận dạng trong thời gian thực (dùng đa luồng)
def capture_and_recognize_video_from_camera():
    def process_video():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không thể mở camera.")
            return

        while True:
            # Ghi lại thời gian bắt đầu
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Lỗi", "Không thể đọc dữ liệu từ camera.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = 0.4
            gray_frame = cv2.resize(gray_frame, None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)  
            best_match = match_currency(gray_frame, template_images)
            end_time = time.time()            
            # Tính thời gian xử lý
            processing_time = end_time - start_time  
            # Tính FPS (Frames Per Second)
            if best_match:
                subfolder, filename = best_match
                cv2.putText(
                    frame,
                    f"Currency: {subfolder} VND",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            # Display FPS on the frame
            cv2.putText(
                frame,
                f"Time: {processing_time:.2f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Camera", frame)

            # Bấm 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Chạy quá trình nhận dạng trong luồng riêng biệt
    threading.Thread(target=process_video).start()

def recognize_from_video_file(file_path):
    def process_video():
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open video.")
            return


        while True:
            # Ghi lại thời gian bắt đầu
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = 0.4
            gray_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            best_match = match_currency(gray_frame, template_images)

            end_time = time.time()            
            # Tính thời gian xử lý
            processing_time = end_time - start_time  
            # Tính FPS (Frames Per Second)
            if best_match:
                subfolder, filename = best_match
                cv2.putText(
                    frame,
                    f"Currency: {subfolder} VND",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            # Display FPS on the frame
            cv2.putText(
                frame,
                f"Time: {processing_time:.2f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            #time.sleep(0.4)  # Reduce frame rate

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=process_video).start()
# Tạo giao diện bằng Tkinter
root = tk.Tk()
root.title("Currency Recognition")

# Đặt kích thước cửa sổ
root.geometry("400x400")

# Khởi tạo thư mục mẫu
root_folder = './dataset_VN_Money'
template_images = load_template_images(root_folder)

# Biến để giữ đường dẫn ảnh đầu vào
input_image_path = None

# Hàm xử lý nút "Chọn ảnh"
def select_image():
    global input_image_path
    default_directory = "./test/images"
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.png")],initialdir=default_directory
    )
    if file_path:
        input_image_path = file_path
        label_image.config(text=f"Ảnh được chọn: {os.path.basename(file_path)}")

# Hàm xử lý nút "Chụp video từ camera"
def record_video_from_camera():
    capture_and_recognize_video_from_camera()

# Hàm xử lý nút "Chọn video"
def select_and_recognize_video():
    default_directory = "./test/videos"
    file_path = filedialog.askopenfilename(
        title="Chọn video", filetypes=[("Video files", "*.mp4;*.avi;*.mkv")],initialdir=default_directory
    )
    if file_path:
        recognize_from_video_file(file_path)

# Tạo khung chứa các nút
button_frame = tk.Frame(root)

# Nút chọn ảnh
button_select_image = tk.Button(
    root, text="Chọn ảnh", command=select_image, width=10, height=2, bg='lightblue'
)
button_select_image.pack(pady=10)

# Nút quay video từ camera
button_record_video = tk.Button(
    root, text="Quay video từ camera", command=record_video_from_camera, width=20, height=2, bg='lightgreen'
)
button_record_video.pack(pady=10)

# Nút chọn video từ tệp
button_select_video = tk.Button(
    root, text="Chọn video", command=select_and_recognize_video, width=10, height=2, bg='orange'
)
button_select_video.pack(pady=10)

# Nhãn hiển thị tên ảnh hoặc video được chọn
label_image = tk.Label(root, text="Chưa có ảnh hoặc video được chọn.", width=25, bg='lightgray')
label_image.pack(pady=10)

# Nút "Xóa" để xóa kết quả hiện tại
def clear_result():
    global input_image_path
    input_image_path = None
    label_image.config(text="Chưa có ảnh hoặc video được chọn.")
button_clear = tk.Button(
    root, text="Xóa", command=clear_result, width=10, height=2, bg='red'
)
button_clear.pack(pady=10)

# Hàm xử lý nút "Start" để bắt đầu nhận dạng từ ảnh đã chọn
def start_recognition():

    if input_image_path:
        # Ghi lại thời gian bắt đầu
        start_time = time.time()
        input_image = cv2.imread(input_image_path)
        scale = 0.4
        input_image = cv2.resize(input_image, None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)  
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        best_match = match_currency(gray_image, template_images)
        end_time = time.time()            
        # Tính thời gian xử lý
        processing_time = end_time - start_time 
        if best_match:
            subfolder, filename = best_match

            cv2.putText(input_image, f"Currency: {subfolder} VND", (10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
            # Display FPS on the frame
            cv2.putText(
                input_image,
                f"Time: {processing_time:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Input Image", input_image)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước khi bắt đầu.")

# Nút bắt đầu quá trình nhận dạng
button_start = tk.Button(
    root, text="Start", command=start_recognition, width=10, height=2, bg='lightgreen'
)
button_start.pack(pady=10)

# Kết thúc việc tạo giao diện Tkinter
root.mainloop()
