# Re-Identification in a Single Feed

**Real-Time Player Detection & Tracking**

## 📌 Description
PlayerTrack detects and tracks players in sports videos using YOLOv11 and DeepSORT, outputting an annotated video for analytics.

## 🚀 Features
- Custom best.pt model for player detection
- DeepSORT tracker for unique IDs
- Real-time video processing
- Generates output video file

## ⚙️ Installation
```bash
git clone <your-repo-url>
cd PlayerTrack
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## ▶️ Usage
Place your input video (`15sec_input_720p.mp4`) in the project folder.
Run:
```bash
python tracker.py
```

Output: `output_tracking.mp4` saved in the same folder.

## 📦 Dependencies
- Python 3.x
- ultralytics (YOLOv11)
- deep_sort_realtime
- opencv-python
- torch

## ⚠️ Notes
- Large files (`best.pt`, videos) are ignored in `.gitignore`.
- Use Git LFS for `.pt` models.

## 👤 Author
Sania

## 📝 License
MIT
