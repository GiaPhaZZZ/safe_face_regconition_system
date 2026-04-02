# Quy trình đăng ký: Nhận vào 1 ảnh + ID người dùng để đặt làm tên folder lưu trữ
# 1. Mediapipe kiểm tra ảnh có khuôn mặt hợp lệ không -> Không thì trả về app cho chụp lại
# 2. Nếu hợp lệ, crop ảnh và trích xuất đặc trưng bằng EdgeFace
# 3. Trả về folder chứa ảnh khuôn mặt.jpg + đặc trưng.npy dùng cho so sánh sau này - gửi tới database lưu trữ


import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from edgeface.backbones import get_model


# CONFIGURATION - cho phép thay đổi
CONFIG = {
    "user_id": "0001",                  # ID người dùng để đặt làm tên folder lưu trữ
    "input_img": "./users/a.jpg",       # Ảnh input của người đăng ký - 1 tấm trực tiếp
    "db_path": "face_database",         # Lưu tạm kết quả vào folder này làm datasebase so sánh (kết quả trả về là ảnh.jpg + đặc trưng.npy)
    "paths": {
        "landmark_task": os.path.join(os.path.dirname(__file__), "face_landmark/face_landmarker.task"),
        "edgeface_weights": os.path.join(os.path.dirname(__file__), "edgeface/edgeface_s_gamma_05.pt"),
    },
    "thresholds": {
        "yaw": 15,                  # Độ nghiêng tối đa hợp lệ
        "pitch": 15,                # Độ ngước lên xuống tối đa hợp lệ
        "min_ratio": 0.10,          # Mặt phải chiếm ít nhất 10% diện tích ảnh
        "max_ratio": 0.50,          # Mặt không được chiếm quá 50% diện tích ảnh
    },
    "dims": {
        "face_size": (112, 112),    # Resize cho nhẹ
        "padding": 0.15
    }
}

# Không nên thay đổi - Do not change
class FaceManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        base_options = python.BaseOptions(model_asset_path=cfg["paths"]["landmark_task"])
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.model = get_model("edgeface_s_gamma_05")
        self.model.load_state_dict(torch.load(cfg["paths"]["edgeface_weights"], map_location=self.device))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def get_embedding(self, face_img):
        """Processes image and returns normalized embedding."""
        img = cv2.resize(face_img, self.cfg["dims"]["face_size"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ((img.astype(np.float32) - 127.5) / 128.0)
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        embedding = self.model(tensor)
        return torch.nn.functional.normalize(embedding, p=2, dim=1).cpu().numpy().flatten()

    def validate_and_crop(self, frame):
        """Checks pose/size and returns cropped face image."""
        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return None, "No face detected"

        matrix = result.facial_transformation_matrixes[0]
        yaw = np.degrees(np.arctan2(matrix[0, 2], matrix[2, 2]))
        pitch = np.degrees(np.arcsin(-matrix[1, 2]))
        
        if abs(yaw) > self.cfg["thresholds"]["yaw"] or abs(pitch) > self.cfg["thresholds"]["pitch"]:
            return None, f"Pose invalid (Yaw:{yaw:.1f}, Pitch:{pitch:.1f})"

        lms = result.face_landmarks[0]
        xs, ys = [m.x * w for m in lms], [m.y * h for m in lms]
        x1, x2, y1, y2 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
        
        ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
        if not (self.cfg["thresholds"]["min_ratio"] < ratio < self.cfg["thresholds"]["max_ratio"]):
            return None, f"Distance invalid (Ratio:{ratio:.2f})"

        pw, ph = int((x2 - x1) * self.cfg["dims"]["padding"]), int((y2 - y1) * self.cfg["dims"]["padding"])
        crop = frame[max(0, y1-ph):min(h, y2+ph), max(0, x1-pw):min(w, x2+pw)]
        return crop, "Success"

    def register_user(self, user_id, img_path):
        
        frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None: return print(f"Error: Could not load {img_path}")

        face_crop, status = self.validate_and_crop(frame)
        if face_crop is None: return print(f"Failed: {status}")

        embedding = self.get_embedding(face_crop)

        user_dir = os.path.join(self.cfg["db_path"], user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        np.save(os.path.join(user_dir, "embedding.npy"), embedding)
        cv2.imwrite(os.path.join(user_dir, "reference.jpg"), face_crop)
        
        print(f"Registered {user_id} successfully in {user_dir}")


# RUN
if __name__ == "__main__":
    manager = FaceManager(CONFIG)
    manager.register_user(CONFIG["user_id"], CONFIG["input_img"])