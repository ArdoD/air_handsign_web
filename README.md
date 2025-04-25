# FSL Video Detection 
 
A Flask-based web application for detecting sign language gestures from uploaded videos using a pre-trained SiameseCNN1D model and MediaPipe for keypoint extraction. 
 
## Prerequisites 
 
- Python 3.8 or higher 
- A trained SiameseCNN1D model (`siamese_cnn1d.pth` file) 
- Support set files (`.npy`) for each class 
- GPU (optional, for faster inference) 
 
## Setup 
 
1. Ensure the trained model file `siamese_cnn1d.pth` is placed in the `models/` directory. 
3. Install dependencies: 
   ```bash 
   pip install -r requirements.txt 
   ``` 
4. Run the application: 
   ```bash 
   python app.py 
   ``` 
5. Open your browser and go to `http://127.0.0.1:5000`. 
 
## Usage 
 
1. On the homepage, upload a video file (supported formats: `.mp4`, `.avi`, `.mov`). 
2. The application will process the video, extract keypoints using MediaPipe, and predict the gesture using the SiameseCNN1D model. 
3. View the predicted gesture and the uploaded video on the result page. 
 
## Project Structure 
 
- `app.py`: Main Flask application. 
- `model.py`: SiameseCNN1D model definition. 
- `process_video.py`: Video processing and keypoint extraction. 
- `static/css/style.css`: CSS styling for the web interface. 
- `templates/index.html`: Homepage for uploading videos. 
- `templates/result.html`: Page for displaying detection results. 
- `support_set/`: Directory for support set `.npy` files. 
- `requirements.txt`: List of Python dependencies. 
- `README.md`: Project documentation. 
 
## Notes 
 
- Ensure the support set `.npy` files match the expected input format (31 frames, 84 features). 
- The application assumes the video contains a single gesture. For multiple gestures, modify `process_video.py` accordingly. 
- Adjust the model path in `app.py` if your `.pth` file is located elsewhere. 
 
## License 
 
This project is licensed under the MIT License. 
