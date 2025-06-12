
# Algorithmic Touch

> A Flask web app for skin disease recognition and sketch-style visualization

## Project Overview

Algorithmic Touch lets users:

1. Upload a photo of a skin lesion  
2. Classify it with a fine-tuned ResNet-34 model into one of seven categories and view the probability distribution  
3. Turn the original image into a clean white-line sketch and display it alongside disease information  

Model weights are hosted on Hugging Face Hub and fetched at runtime with `hf_hub_download`. The frontend uses p5.js to render a soft, full-screen animated background.

## Demo

<div align="center">
<img src="./docs/screenshot-1.gif" width="45%" />  
<img src="./docs/screenshot-2.gif" width="45%" />

*Left: upload page · Right: result page*

</div>

## Installation & Running

### Prerequisites

- Python ≥ 3.8  
- Git  
- Internet access to Hugging Face Hub

### Clone the Repo

```bash
git clone https://github.com/littlexx15/Algorithmic-Touch.git
cd AlgorithmicTouch
```


### Install Dependencies

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### Set Up Authentication

You need an HF_TOKEN to fetch the model from Hugging Face Hub.

Mac：

```bash
export HF_TOKEN="your_huggingface_token"
```

Windows PowerShell：

```bash
$env:HF_TOKEN = "your_huggingface_token"
```

### Start the Server

```bash
python app.py
```

The app will run on 0.0.0.0:8000. Open your browser to http://localhost:8000



