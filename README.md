# Med-ML

Streamlit app for training and using a simple CNN to classify chest X-ray images.

## Quick Start (Local)
1. Install Python 3.10+.
2. In this folder, run:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run xray_cnn_trainer.py
   ```

## Train a Model
1. Prepare one ZIP file per class (example: `Covid.zip`, `Normal.zip`, `Pneumonia.zip`).
   - Each ZIP should contain images only for that class.
2. In the app, upload all class ZIPs under **Upload Folders for Training**.
3. Click **Train CNN** and wait for training to finish.
4. Download:
   - **xray_cnn.pth** (model + labels)
   - **labels.json** (label map, optional)

## Predict (Use a Trained Model)
1. Under **Upload Trained Model for Prediction**, upload `xray_cnn.pth`.
2. (Optional) Upload `labels.json` if you trained elsewhere.
3. Upload one or more images under **Upload images to predict**.
4. Click **Predict** to see class names.

## Notes
- The app runs training in your browser session, so large datasets may be slow.
- For best results, keep image sizes consistent and use clear, labeled folders.
