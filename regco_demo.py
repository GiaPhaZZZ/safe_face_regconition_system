import os
import cv2
import torch
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from anti_spoofing.anti_fake import test
from edgeface.backbones import get_model


# CONFIGURATION - Cho phép thay đổi
CONFIG = {
    "paths": {
        "input_image": "./users/b.jpg",     #Thay đổi input ảnh tùy ý
        "database_dir": "face_database",    # Đang dùng tạm folder này làm datasebase lấy so sánh
        
        "anti_spoof_model": os.path.join(os.path.dirname(__file__), "anti_spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"),
        "mediapipe_task": os.path.join(os.path.dirname(__file__), "face_landmark/face_landmarker.task"),
        "edgeface_weights": os.path.join(os.path.dirname(__file__), "edgeface/edgeface_s_gamma_05.pt"),
    },
    "thresholds": {
        "yaw": 15,                  # Độ nghiêng tối đa hợp lệ
        "pitch": 15,                # Độ ngước lên xuống tối đa hợp lệ
        "similarity": 0.5,          # Ngưỡng tương đồng tối thiểu để xác định danh tính khuôn mặt
        "min_face_ratio": 0.10,     # Mặt phải chiếm ít nhất 10% diện tích ảnh
        "max_face_ratio": 0.50,     # Mặt không được chiếm quá 50% diện tích ảnh
    },
    "dimensions": {
        "edgeface_size": (112, 112),    # Resize cho nhẹ
        "crop_padding": 0.15            # Pad thêm viền
    }
}


# Không nên thay đổi - Do not change
class FaceSystem:
    def __init__(self, config):
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load MediaPipe
        base_options = python.BaseOptions(model_asset_path=self.cfg["paths"]["mediapipe_task"])
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # 2. Load EdgeFace
        self.face_engine = get_model("edgeface_s_gamma_05")
        self.face_engine.load_state_dict(torch.load(self.cfg["paths"]["edgeface_weights"], map_location=self.device))
        self.face_engine.to(self.device).eval()

    @torch.no_grad()
    def get_embedding(self, face_img):
        face_img = cv2.resize(face_img, self.cfg["dimensions"]["edgeface_size"])
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = ((face_img.astype(np.float32) - 127.5) / 128.0)
        face_img = np.transpose(face_img, (2, 0, 1))
        tensor = torch.from_numpy(face_img).unsqueeze(0).to(self.device)
        
        embedding = self.face_engine(tensor)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()

    def validate_and_extract(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.detector.detect(mp_image)
        
        # --- MediaPipe Check ---
        if not result.face_landmarks:
            print("[MediaPipe] FAILED: No face detected in frame.")
            return False, None

        # Euler Angles Check
        matrix = result.facial_transformation_matrixes[0]
        yaw = np.degrees(np.arctan2(matrix[0, 2], matrix[2, 2]))
        pitch = np.degrees(np.arcsin(-matrix[1, 2]))

        if abs(yaw) > self.cfg["thresholds"]["yaw"] or abs(pitch) > self.cfg["thresholds"]["pitch"]:
            print(f"[MediaPipe] FAILED: Face tilted too much (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})")
            return False, None

        # Face Size (Ratio) Check
        landmarks = result.face_landmarks[0]
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]
        x_min, x_max, y_min, y_max = int(min(x_coords)), int(max(x_coords)), int(min(y_coords)), int(max(y_coords))
        
        face_ratio = ((x_max - x_min) * (y_max - y_min)) / (h * w)
        if not (self.cfg["thresholds"]["min_face_ratio"] < face_ratio < self.cfg["thresholds"]["max_face_ratio"]):
            print(f"[MediaPipe] FAILED: Face distance incorrect (Ratio: {face_ratio:.2f})")
            return False, None

        print(f"\n[MediaPipe] SUCCESS: Face validated (Yaw: {yaw:.1f}, Pitch: {pitch:.1f})\n")

        # In-Memory Crop
        pad = self.cfg["dimensions"]["crop_padding"]
        pad_w, pad_h = int((x_max - x_min) * pad), int((y_max - y_min) * pad)
        fx1, fy1 = max(0, x_min - pad_w), max(0, y_min - pad_h)
        fx2, fy2 = min(w, x_max + pad_w), min(h, y_max + pad_h)
        
        return True, frame[fy1:fy2, fx1:fx2]

def recognize_face(input_embedding, db_path, threshold):
    best_match_id, highest_sim = "Unknown", -1.0
    if not os.path.exists(db_path): return best_match_id, 0.0

    for user_id in os.listdir(db_path):
        emb_file = os.path.join(db_path, user_id, "embedding.npy")
        if os.path.exists(emb_file):
            db_embedding = np.load(emb_file)
            similarity = np.dot(input_embedding, db_embedding)
            if similarity > highest_sim:
                highest_sim = similarity
                best_match_id = user_id

    if highest_sim >= threshold:
        return best_match_id, highest_sim
    return "Unknown", highest_sim


# MAIN EXECUTION - Thay đổi kết quả trả về tùy ý, hiện tại chỉ là print ra thông báo
def main():

    frame = cv2.imread(CONFIG["paths"]["input_image"])
    if frame is None:
        print("\nError: Could not read input image.")
        return

    system = FaceSystem(CONFIG)

    # 1. Validation
    success, face_crop = system.validate_and_extract(frame)

    if success:
        # 2. Anti-Spoofing
        temp_path = "temp_crop.jpg"
        cv2.imwrite(temp_path, face_crop)
        label = test(temp_path, CONFIG["paths"]["anti_spoof_model"], device_id=0)
        os.remove(temp_path) # Clean up immediately
        
        if label == 1:
            print("\n[Anti-Spoof] SUCCESS: Real face detected.")
            
            # 3. Recognition
            input_emb = system.get_embedding(face_crop)
            user_id, score = recognize_face(
                input_emb, 
                CONFIG["paths"]["database_dir"], 
                CONFIG["thresholds"]["similarity"]
            )
            
            if user_id != "Unknown":
                print(f"\n[Identity] MATCH: {user_id} (Score: {score:.4f})\n")
            else:
                print(f"\n[Identity] NO MATCH: Closest was {score:.4f}\n")
        else:
            print("\n[Anti-Spoof] ALERT: Spoof attack detected!")
    else:
        print("\nSystem stopped at MediaPipe validation.")

if __name__ == "__main__":
    main()