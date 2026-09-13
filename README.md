# Satellite Change Detection

A Streamlit application that compares aligned before-and-after RGB images and highlights predicted building changes.

The model is a Siamese ResNet-18 U-Net trained on LEVIR-CD in Google Colab. Both images pass through a shared encoder, and differences between their features are used to predict a binary change mask.

## Features

- Upload before-and-after PNG or JPEG images.
- View the predicted mask and a red overlay.
- Adjust the prediction threshold.
- Download masks, overlays, probability images, and a JSON summary.
- Run locally on CPU without retraining.

## Test results

Evaluated on the 128 held-out LEVIR-CD test pairs at threshold 0.50:

| Metric | Result |
|---|---:|
| Precision | 89.78% |
| Recall | 87.64% |
| F1 | 88.70% |
| IoU | 79.69% |
| Pixel accuracy | 98.86% |

Training stopped at epoch 18 after eight epochs without validation F1 improvement. Evaluation used the best validation checkpoint.

Rare intermediate grayscale values in the dataset masks were thresholded to binary values before training and evaluation. These results reflect that preprocessing.

## Run locally on Windows

Install Python 3.12 and Git.

### 1. Download the code

```text
git clone https://github.com/RoyalAskarov/satellite-change-detection.git
cd satellite-change-detection
```

Alternatively, download and extract the repository ZIP from GitHub.

### 2. Download the trained model

Download `levir_cd_best.pt` from the project's [GitHub Releases](https://github.com/RoyalAskarov/satellite-change-detection/releases).

Place it in the main project folder beside `app.py`. Do not rename it.

The model is distributed separately from the Git repository. No training dataset is required to run the app.

### 3. Create a Python environment

```text
py -3.12 -m venv .venv
```

### 4. Install packages

Install CPU-only PyTorch first:

```text
.\.venv\Scripts\python.exe -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the application requirements:

```text
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
```

`requirements-lock.txt` records the package versions from the author's working Windows environment. `colab_environment.txt` records the separate training environment; it is not needed to run the app.

### 5. Start the application

```text
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the local address printed in the terminal, usually `http://localhost:8501`.

Keep the terminal running. Press Ctrl+C to stop the app.

## Using the app

1. Upload aligned images of the same location from two dates.
2. Keep the threshold at 0.50 for the setting used in test evaluation.
3. Click **Detect building changes**.
4. Inspect and download the results.

The changed-pixel percentage is not an accuracy score, building count, or ground-area measurement.

## Customize the interface

Open the project folder in PyCharm and select the `.venv` interpreter.

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface, controls, and result display |
| `model.py` | Model architecture and image normalization |
| `inference.py` | Patch-based prediction |
| `training.py` | Training and evaluation utilities |
| `test_metrics.json` | Measured test results |
| `history.json` | Training and validation history |

For changes to titles, layout, colors, and controls, start with `app.py`. UI changes do not require retraining. Changes to the architecture or preprocessing require additional compatibility checks.

## Contributing

1. Fork this repository.
2. Create a branch for your changes.
3. Run the app locally and test your changes.
4. Submit a pull request describing what changed.
5. Include screenshots for UI changes.

Do not commit virtual environments, credentials, datasets, or model checkpoints.

## Limitations and dataset attribution

This model predicts building-related changes, including additions and removals, without distinguishing between them. It does not detect deforestation or register misaligned images.

Use aligned RGB imagery at a scale similar to LEVIR-CD, approximately 0.5 metres per pixel. Performance on other sensors, resolutions, or regions has not been established. Predictions may contain missed changes and false positives.

The LEVIR-CD authors restrict dataset images and annotations to academic use and prohibit their commercial use. See the [official dataset page](https://github.com/justchenhao/LEVIR). Sharing this project does not override those terms.

Dataset reference:

Chen, H. and Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. Remote Sensing, 12(10), 1662. https://doi.org/10.3390/rs12101662