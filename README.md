# 🎭 TableTalk — Narrative Voice Processing & Classification

> **Technical Assessment Submission**  
> Dataset: [RAVDESS Emotional Speech Audio](https://zenodo.org/record/1188976)  
> Accuracy Achieved: **95.14%** (MLP) | All 5 Tasks Complete ✅

---

## 📋 Project Structure

```
tabletalk-ravdess/
├── tabletalk-human-ai-technical-test.ipynb   # Main Kaggle notebook (all tasks)
├── README.md                                  # This file
└── TableTalk_Technical_Report.docx           # 4-page technical report
```

---

## 🚀 Running on Kaggle (Recommended)

This notebook is designed to run on Kaggle with GPU acceleration. Follow these steps:

### Step 1 — Add the Dataset
1. Open the notebook on Kaggle
2. Click **+ Add Data** (top right)
3. Search for: `ravdess-emotional-speech-audio` by `uwrfkaggler`
4. Click **Add**

### Step 2 — Enable GPU
1. Go to **Settings** (right sidebar)
2. Under **Accelerator**, select **GPU T4 x2** or **GPU P100**
3. Enable **Internet** (required for pip installs)

### Step 3 — Run All Cells
Click **Run All** or press `Shift + Enter` on each cell sequentially.

> ⏱️ **Expected runtime:** ~6–7 minutes on GPU

### Expected Output Per Task
| Task | Output |
|------|--------|
| Task 1 | `ravdess_features.csv` + waveform/spectrogram plots |
| Task 2 | Confusion matrix + model comparison chart (95.14% accuracy) |
| Task 3 | `transcription_results.csv` + WER = 16.7% |
| Task 4 | Live retrieval query results printed to console |
| Bonus | Storytelling feature boxplots |

---

## 💻 Running Locally

### Requirements
- Python 3.9+
- pip
- ~4 GB RAM (for feature extraction)
- GPU optional (CPU works, slower training)

### Step 1 — Clone the repo
```bash
git clone https://github.com/your-username/tabletalk-ravdess.git
cd tabletalk-ravdess
```

### Step 2 — Install dependencies
```bash
pip install librosa openai-whisper scikit-learn pandas numpy \
            matplotlib seaborn tqdm torch jiwer jupyter
```

> ⚠️ `openai-whisper` requires `ffmpeg` installed on your system:
> - **Ubuntu/Debian:** `sudo apt install ffmpeg`
> - **Mac:** `brew install ffmpeg`
> - **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Step 3 — Download RAVDESS Dataset
Download from [Zenodo](https://zenodo.org/record/1188976) and extract so the folder structure looks like:

```
data/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   └── ...
├── Actor_02/
└── ...
```

### Step 4 — Update the dataset path
Open the notebook and change `DATA_PATH` in the **Dataset Loading** cell:

```python
# Change this line:
DATA_PATH = '/kaggle/input/datasets/uwrfkaggler/ravdess-emotional-speech-audio'

# To your local path:
DATA_PATH = './data'
```

### Step 5 — Launch Jupyter and run
```bash
jupyter notebook tabletalk-human-ai-technical-test.ipynb
```

Then click **Kernel → Restart & Run All**.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `librosa` | ≥ 0.10 | Audio loading, feature extraction |
| `openai-whisper` | latest | Speech transcription (Task 3) |
| `torch` | ≥ 2.0 | MLP training (Task 2) |
| `scikit-learn` | ≥ 1.3 | SVM, RF, Logistic Regression |
| `pandas` / `numpy` | latest | Data manipulation |
| `matplotlib` / `seaborn` | latest | Visualisations |
| `tqdm` | latest | Progress bars |
| `jiwer` | latest | Word Error Rate calculation |

---

## 🧠 Model Summary

```
EmotionMLP Architecture:
  Input  (461) → Linear(512) → BatchNorm → ReLU → Dropout(0.35)
               → Linear(256) → BatchNorm → ReLU → Dropout(0.30)
               → Linear(128) → BatchNorm → ReLU → Dropout(0.20)
               → Linear(8)   [output: 8 emotion classes]

Optimizer : Adam (lr=0.001, weight_decay=1e-4)
Scheduler : ReduceLROnPlateau (patience=10, factor=0.5)
Epochs    : 150  |  Batch size: 64  |  Split: Stratified 80:20
```

---

## 📊 Results Summary

| Task | Metric | Result |
|------|--------|--------|
| Task 1 | Features extracted | 461-dim vector, 2880 samples |
| Task 2 | MLP Accuracy | **95.14%** |
| Task 2 | MLP F1-Score (weighted) | **95.18%** |
| Task 3 | Whisper WER | **16.7%** (83.3% correct) |
| Task 4 | Retrieval queries | 5 example queries working |
| Bonus  | Features identified | 5 storytelling features |

---

## 📁 Output Files Generated

After running the notebook, these files will be saved:

```
ravdess_features.csv          # 461-feature dataset (Task 1)
transcription_results.csv     # Whisper transcripts + WER (Task 3)
dataset_distribution.png      # Emotion distribution charts
audio_visualization.png       # Waveform + MFCC + Mel spectrogram
classification_results.png    # Training curve + confusion matrix
model_comparison.png          # Model accuracy bar chart
storytelling_analysis.png     # Bonus feature boxplots
```

---

## ⚠️ Troubleshooting

**`librosa` import error:**
```bash
pip install --upgrade librosa soundfile
```

**Whisper model download slow:**  
Whisper downloads the model on first run (~460 MB for `small`). This requires internet access — make sure internet is enabled on Kaggle.

**CUDA out of memory:**  
Reduce batch size in the MLP training cell:
```python
train_loader = DataLoader(train_dataset, batch_size=32, ...)  # reduce from 64
```

**`jiwer` not found:**
```bash
pip install jiwer
```

---

## 📄 License

This project is submitted as part of the TableTalk Human-AI Technical Assessment. The RAVDESS dataset is licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
