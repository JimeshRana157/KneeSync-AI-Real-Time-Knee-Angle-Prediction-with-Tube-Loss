# KneeSync AI – Real‑Time Knee Angle Prediction with Tube Loss

This project reproduces and improves the CNN‑LSTM transfer learning framework for **real‑time knee angle prediction** from 4‑channel surface EMG signals, as described in **Mollahossein et al. (2024)**.  
It introduces **Tube Loss** (Anand et al., 2025) to provide **calibrated prediction intervals** alongside point estimates.

> **🎯 Objective**  
> Predict the knee joint angle **10 ms ahead** using EMG from the biceps femoris, rectus femoris, semitendinosus, and vastus medialis muscles.  
> Transfer knowledge from a large healthy‑subject dataset (Georgia Tech) to **normal** and **pathological** patients (UCI) with only **3–7 gait cycles** of fine‑tuning.

---

## ✨ Key Features

1. **Two Model Configurations**  
   - **SIC** (Single‑Input Configuration) – EMG only  
   - **DIC** (Dual‑Input Configuration) – EMG + historical knee angle with attention fusion

2. **Tube Loss Integration**  
   - Simultaneously learns lower (μ₁) and upper (μ₂) bounds of a **prediction interval** with asymptotic coverage guarantee at *t* = 0.90.  
   - A small MSE component anchors the midpoint, preserving point accuracy.  
   - Post‑hoc **conformal calibration** (optional) can further tighten intervals.

3. **3‑Stage Transfer Protocol**  
   - **Pre‑training** on Georgia Tech (425k windows)  
   - **Stage‑1** population adaptation (leave‑one‑out across 11 training subjects)  
   - **Stage‑2** subject‑specific fine‑tuning (50/50 split)

4. **Live Demo**  
   - A browser‑based demo (`index.html`) connects to a FastAPI backend serving the best DIC + Tube Loss checkpoint.  
   - Users can simulate muscle activation via sliders and see real‑time predictions and prediction intervals.

---

## 📊 Results at a Glance

| Configuration | Setting | Mean NMAE | Mean R² | Mean PICP (target 0.90) | Mean NMPIW |
|---------------|---------|-----------|---------|--------------------------|------------|
| SIC (MSE)     | Normal  | 10.58 %   | 0.649   | –                        | –          |
| SIC (MSE)     | Abnormal| 7.18 %    | 0.808   | –                        | –          |
| **DIC (MSE)** | Normal  | **1.60 %**| **0.986**| –                      | –          |
| **DIC (MSE)** | Abnormal| **0.81 %**| **0.998**| –                      | –          |
| DIC + Tube Loss | Normal | 1.99 %   | 0.983   | **0.916**                | 8.15 %     |
| DIC + Tube Loss | Abnormal| 1.05 %   | 0.997   | **0.942**                | **5.33 %** |

- **DIC Abnormal** (MSE) **3.5× better** than the original paper (Subject 11).  
- **DIC + Tube Loss** delivers reliable intervals with minimal point accuracy loss. The best result: **PICP = 0.942, NMPIW = 5.33%, NMAE = 1.05%** on abnormal subjects.  
- Full evaluation (11 subjects each) is in `knee_tube_loss_v2.ipynb`.

1. Clone the repository
```bash
git clone https://github.com/YourUsername/kneesync-ai.git
cd kneesync-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the datasets
- **Georgia Tech (Source Domain)**  
  22 healthy subjects, MATLAB `.mat` files.  
  *Example source: Camargo et al. dataset (public domain).*
- **UCI Lower Limb (Target Domain)**  
  Normal & Abnormal walking `.txt` files.  
  [UCI Repository](https://archive.ics.uci.edu/dataset/788/lower+limb+emg)

Place the data in the expected folders and adjust paths in the notebook (Cell 2).

### 4. (Optional) Place pretrained checkpoints
If you want to skip training, place the provided `.pt` files in `checkpoints/`.

---

## 🖥️ Running the Notebook

Open `knee_tube_loss_v2.ipynb` in Jupyter, VS Code, or Google Colab.  
The notebook is fully self‑contained and runs the entire pipeline:

- **Cells 1–8**: Setup, model definition, preprocessing utilities.  
- **Cells 9–11**: Georgia Tech data loading and pretraining (SIC).  
- **Cells 12–20**: UCI transfer learning for both SIC & DIC, with leave‑one‑out evaluation.  
- **Cells 21–22**: Results summary and visualisation.  
- **Cells 23–24**: Conformal calibration (optional post‑processing).

The training will generate output checkpoints in `./checkpoints_tube/`.  
Training on a GPU (e.g., Colab free tier) is recommended – the notebook auto‑detects CUDA.

---

## 🌐 Live Demo (Frontend + Backend)

The live demo shows real‑time knee angle prediction using the **best DIC + Tube Loss checkpoint**.

### Backend (FastAPI)
1. Upload `server.py` and the checkpoint `s2_dic_abnormal_loo10.pt` to Google Colab.
2. Install required packages:
   ```bash
   !pip install fastapi uvicorn pyngrok
   ```
3. Set your ngrok authtoken (free from [ngrok.com](https://ngrok.com)).
4. Run `server.py` in a background thread:
   ```python
   import threading, time, uvicorn
   from pyngrok import ngrok

   ngrok.set_auth_token("YOUR_REAL_TOKEN")
   ngrok.kill()
   time.sleep(1)

   def start():
       uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")

   t = threading.Thread(target=start, daemon=True)
   t.start()
   time.sleep(5)

   public_url = ngrok.connect(8000).public_url
   print("API ready at:", public_url + "/predict")
   ```
5. The cell will print a public URL like `https://xxx.ngrok-free.dev`.

### Frontend
1. Open `index.html` in any browser.
2. Replace the `apiUrl` inside the `<script>` tag with your ngrok URL (keep `/predict` at the end).
3. Use the sliders or click a **preset** (Swing / Stance / Stair Climb) – the chart updates automatically with predictions from the real model.
