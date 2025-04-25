import cv2
import mediapipe as mp
import numpy as np

def process_video_keypoints(video_path, max_value=1080.0):
    print(f"Opening video: {video_path}")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        raise ValueError("Cannot open video file")

    keypoints_list = []
    frame_count = 0
    last_valid_keypoints = None  # Simpan frame valid terakhir untuk padding
    max_frames = 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video after {frame_count} frames")
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Dapatkan dimensi frame untuk konversi ke piksel
        h, w, _ = frame.shape

        # Inisialisasi keypoint per frame dengan nol (84 fitur)
        frame_keypoints = np.zeros(84, dtype=np.float32)

        valid_frame = False  # Flag untuk frame dengan gerakan tangan
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            print(f"Hand detected in frame {frame_count}, number of hands: {num_hands}")

            # Ambil keypoint dari tangan yang terdeteksi
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= 2:  # Batasi hanya dua tangan
                    break
                # Setiap tangan akan mengisi 42 fitur (21 landmark x 2 koordinat: x, y saja)
                keypoints = []
                for lm in hand_landmarks.landmark:
                    x = lm.x * w  # Konversi ke piksel
                    y = lm.y * h  # Konversi ke piksel
                    keypoints.append([x, y])
                keypoints = np.array(keypoints, dtype=np.float32).flatten()  # Shape: (42,)
                # Periksa apakah keypoints valid (bukan semua nol)
                if not np.all(keypoints == 0):
                    start_idx = hand_idx * 42
                    frame_keypoints[start_idx:start_idx + 42] = keypoints
                    valid_frame = True

            if valid_frame:
                keypoints_list.append(frame_keypoints)
                last_valid_keypoints = frame_keypoints.copy()  # Simpan frame valid terakhir
                print(f"Frame {frame_count} keypoints length: {len(frame_keypoints)}")

        if len(keypoints_list) >= max_frames:
            print("Reached 30 frames with keypoints")
            break

    cap.release()
    hands.close()

    # Jika kurang dari 30 frame, lakukan padding
    if len(keypoints_list) > 0:
        if len(keypoints_list) < max_frames:
            print(f"Only {len(keypoints_list)} valid frames available, padding with last valid frame to {max_frames}")
            while len(keypoints_list) < max_frames:
                if last_valid_keypoints is not None:
                    keypoints_list.append(last_valid_keypoints.copy())
                else:
                    print("No valid frame for padding, using zeros")
                    keypoints_list.append(np.zeros(84, dtype=np.float32))
    else:
        print(f"Failed: No frames with hand movement in {video_path}")
        keypoints_list = [np.zeros(84, dtype=np.float32)] * max_frames

    # Konversi ke array NumPy
    try:
        keypoints_array = np.array(keypoints_list[:max_frames], dtype=np.float32)
        print(f"Final keypoints shape: {keypoints_array.shape}")
    except Exception as e:
        print(f"Error creating keypoints array: {str(e)}")
        print("Keypoints list contents:", keypoints_list)
        raise

    # Normalisasi keypoint (sama seperti kode normalisasi)
    keypoints_array = keypoints_array / max_value

    return keypoints_array