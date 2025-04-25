import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import numpy as np
from model import SiameseCNN1D
from process_video import process_video_keypoints
from sklearn.metrics.pairwise import euclidean_distances
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SUPPORT_SET_DIR'] = r"C:\Users\bcamaster\Documents\Ardo\AIR-Handsign\Web\fsl_video_detection\support_set"
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Batas ukuran file 50MB
app.secret_key = "super_secret_key"

# Pastikan folder upload dan support set ada
for folder in [app.config['UPLOAD_FOLDER'], app.config['SUPPORT_SET_DIR']]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Muat model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseCNN1D(input_channels=84, embedding_size=128)  # Inisialisasi model
try:
    model.load_state_dict(
        torch.load(
            r"C:\Users\bcamaster\Documents\Ardo\AIR-Handsign\Model\siamese_cnn1d_1layer_mediapipe_KataHurufAngka_hardtriplet_margin2_epoch300_try9.pth",
            map_location=device,
            weights_only=True
        )
    )
except Exception as e:
    print(f"Error loading model weights: {e}")
    raise
model.to(device)
model.eval()

# Muat support set dengan batasan jumlah per kelas
support_embeddings = []
support_labels = []
classes = ["Aku", "Semangat", "Cinta", "Kenapa", "Tidak", "Kecewa", "A", "B", "C", "D", "E", "F", "J", "M", "N", "O", "P", "Q", "R", "W", "T", "1", "2", "3","10","Mau", "Bicara", "Kamu", "TerimaKasih", "U", "V","I", "X", "Y", "Z", "4","5", "6", "7","Sedih", "Maaf", "Sabar", "Sayang", "Senang", "H", "S","G" , "K", "L", "8", "9"]
max_support_per_class = 5  # Tentukan jumlah maksimal support set per kelas (ubah sesuai kebutuhan)

for cls in classes:
    cls_dir = os.path.join(app.config['SUPPORT_SET_DIR'], cls)
    if not os.path.exists(cls_dir):
        print(f"Warning: Directory for class {cls} not found at {cls_dir}")
        continue

    # Cari semua file .npy di dalam folder kelas
    npy_files = [f for f in os.listdir(cls_dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"Warning: No .npy files found for class {cls} in {cls_dir}")
        continue

    # Batasi jumlah file yang diproses per kelas
    if len(npy_files) > max_support_per_class:
        npy_files = random.sample(npy_files, max_support_per_class)
        print(f"Class {cls}: Limited to {max_support_per_class} support set files: {npy_files}")
    else:
        print(f"Class {cls}: Using all {len(npy_files)} support set files: {npy_files}")

    for npy_file in npy_files:
        file_path = os.path.join(cls_dir, npy_file)
        try:
            data = np.load(file_path)
            # Pastikan data memiliki bentuk yang benar
            if data.ndim == 2 and data.shape[0] == 30 and data.shape[1] == 84:
                data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            elif data.ndim == 3 and data.shape[1] == 30 and data.shape[2] == 84:
                data = data[0]  # Ambil sampel pertama jika ada banyak
                data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            else:
                print(f"Warning: Invalid shape for file {npy_file} in class {cls}: {data.shape}. Expected (30, 84) or (N, 30, 84)")
                continue

            print(f"File {npy_file} for class {cls}: data_tensor shape = {data_tensor.shape}")
            with torch.no_grad():
                embedding = model(data_tensor).cpu().numpy()
            support_embeddings.append(embedding)
            support_labels.append(cls)
        except Exception as e:
            print(f"Error loading file {npy_file} for class {cls}: {str(e)}")
            continue

if not support_embeddings:
    raise ValueError("No valid support set data loaded. Please check .npy files.")

support_embeddings = np.concatenate(support_embeddings, axis=0)
support_labels = np.array(support_labels)
print(f"Total support embeddings: {support_embeddings.shape[0]} samples")

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    print(f"Checking file: {filename}")
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    print("Received request:", request.method)
    if request.method == 'POST':
        print("Files in request:", request.files)
        if 'file' not in request.files:
            print("No file part in request")
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print("File received:", file.filename)
        if file.filename == '':
            print("No selected file")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print("File allowed, saving as:", filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print("File saved at:", filepath)
            try:
                # Proses video untuk mendapatkan keypoint
                print("Processing video keypoints...")
                keypoints = process_video_keypoints(filepath)
                if keypoints.shape != (30, 84):  # Sesuaikan dengan 30 frame
                    raise ValueError(f"Invalid keypoints shape: {keypoints.shape}. Expected (30, 84)")
                keypoints_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(device)
                print(f"Video keypoints_tensor shape = {keypoints_tensor.shape}")

                # Dapatkan embedding dari video
                print("Generating video embedding...")
                with torch.no_grad():
                    video_embedding = model(keypoints_tensor).cpu().numpy()
                print("Video embedding generated")

                # Bandingkan dengan support set menggunakan jarak Euclidean
                print("Computing distances...")
                distances = euclidean_distances(video_embedding, support_embeddings)
                nearest_idx = np.argmin(distances, axis=1)
                predicted_class = support_labels[nearest_idx[0]]
                print(f"Predicted class: {predicted_class}")

                # Kirim hasil ke halaman result
                print("Rendering result page")
                return render_template('result.html', prediction=predicted_class, video_file=filename)
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                flash(f"Error processing video: {str(e)}")
                return redirect(request.url)
            finally:
                if os.path.exists(filepath):
                    print("Cleaning up: removing file", filepath)
                    os.remove(filepath)
        else:
            print("Invalid file format")
            flash('Invalid file format. Allowed formats: mp4, avi, mov')
            return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)