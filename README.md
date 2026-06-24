# STM32 Edge AI

An end-to-end pipeline to train machine learning models on your PC and deploy them to STM32 microcontrollers.

![Pipeline](docs/images/pipeline.svg)

## Why this project exists
STM32 microcontrollers power smart devices, sensors, and automation systems. This repository helps beginners and intermediate users move from dataset to embedded deployment with a single pipeline.

You will learn how to:
- build and train ML models using TensorFlow/Keras
- preprocess CSV and structured data for small MCU models
- select STM32-friendly architectures like MLP, CNN, DS-CNN, Conv1D, and LSTM
- quantize models to full-int8 TFLite for embedded inference
- convert TFLite models into C arrays for STM32 projects
- deploy using TensorFlow Lite for Microcontrollers or STM32Cube.AI

## What makes this project publishing ready
- modular Python scripts for data, training, quantization, and deployment
- beginner-friendly documentation, guides, and FAQs
- publishing checklist and contribution guide
- GitHub Actions workflow for syntax validation
- STM32 deployment notes and ready-to-use C conversion helper
- example synthetic dataset and demo pipeline

## Quick start

### 1. Clone the repository
```bash
git clone https://github.com/HarshRajTiwary/STM32_Edge_AI.git
cd STM32_Edge_AI
```

### 2. Create a Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

### 4. Generate a demo dataset
```bash
.venv/bin/python src/generate_synthetic.py --out data/synthetic.csv --repr data/synthetic_repr.csv --samples 1000 --features 10
```

### 5. Train a model
```bash
.venv/bin/python -m src.train --data data/synthetic.csv --target target --epochs 10 --out outputs
```

### 6. Quantize the model
```bash
.venv/bin/python -m src.quantize --saved_model outputs/saved_model --tflite outputs/model_int8.tflite --repr-csv data/synthetic_repr.csv --target target
```

### 7. Convert to STM32 C source
```bash
.venv/bin/python src/deploy/convert_to_c.py outputs/model_int8.tflite stm32/model_data.c model_data
```

## Detailed workflow

### Step 1: Prepare your data
Put your CSV files inside the `data/` folder.
- For regression or classification, use a target column called `target` or similar.
- Tabular input columns must be numeric.
- If you want to use image or sequence data, reshape the input using `--reshape`.

### Step 2: Preprocess data
The pipeline automatically scales numeric features and saves the scaler for future inference.
- `src/data.py` loads CSV data and splits it into training and test sets.
- `src/preprocess.py` fits a scaler and saves it to `outputs/scaler.joblib`.

### Step 3: Choose a model architecture
Supported models:
- `mlp` — dense neural network for tabular data
- `cnn2d` — 2D convolutional network for images and 2D sensor grids
- `ds_cnn` — depthwise separable CNN for smaller footprint and faster MCU inference
- `conv1d` — 1D convolutional network for time-series or audio signals
- `lstm` — sequence model for time-dependent data

### Step 4: Train the model
Use `src/train.py` with the `--model` flag.
```bash
.venv/bin/python -m src.train --data data/synthetic.csv --target target --model mlp --epochs 20 --out outputs
```
For CNN/LSTM/Conv1D models, include `--reshape`:
```bash
.venv/bin/python -m src.train --data data/mydata.csv --target label --model cnn2d --reshape 28,28,1 --epochs 20 --out outputs
```

### Step 5: Quantize the model
Convert the saved model to a TFLite file optimized for MCU inference.
```bash
.venv/bin/python -m src.quantize --saved_model outputs/saved_model --tflite outputs/model_int8.tflite --repr-csv data/synthetic_repr.csv --target target
```

### Step 6: Deploy to STM32
Convert the `.tflite` file into a C array so the model can be embedded in STM32 firmware.
```bash
.venv/bin/python src/deploy/convert_to_c.py outputs/model_int8.tflite stm32/model_data.c model_data
```

Then follow the deployment guide:
- `stm32/README_STM32.md`
- `docs/STM32_DEPLOYMENT_GUIDE.md`

![Model types](docs/images/model-types.svg)

## Example commands

### Train a tabular MLP
```bash
.venv/bin/python -m src.train --data data/mydata.csv --target target --model mlp --epochs 20 --out outputs
```

### Train a small 2D CNN for image data
```bash
.venv/bin/python -m src.train --data data/mydata.csv --target target --model cnn2d --reshape 28,28,1 --epochs 20 --out outputs
```

### Train a small DS-CNN for embedded image inference
```bash
.venv/bin/python -m src.train --data data/mydata.csv --target target --model ds_cnn --reshape 28,28,1 --epochs 20 --out outputs
```

### Train a time-series model with Conv1D
```bash
.venv/bin/python -m src.train --data data/mydata.csv --target target --model conv1d --reshape 100,1 --epochs 20 --out outputs
```

### Train an LSTM sequence model
```bash
.venv/bin/python -m src.train --data data/mydata.csv --target target --model lstm --reshape 100,1 --epochs 20 --out outputs
```

### Quantize and convert
```bash
.venv/bin/python -m src.quantize --saved_model outputs/saved_model --tflite outputs/model_int8.tflite --repr-csv data/mydata_sample.csv --target target
.venv/bin/python src/deploy/convert_to_c.py outputs/model_int8.tflite stm32/model_data.c model_data
```

## Repository structure

```text
STM32_Edge_AI/
├── README.md
├── requirements.txt
├── LICENSE
├── docs/
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   ├── FAQ.md
│   ├── OVERVIEW.md
│   ├── PROJECT_CHECKLIST.md
│   ├── STM32_DEPLOYMENT_GUIDE.md
│   └── images/
│       ├── pipeline.svg
│       └── model-types.svg
├── data/
├── outputs/
├── src/
│   ├── data.py
│   ├── generate_synthetic.py
│   ├── model.py
│   ├── preprocess.py
│   ├── quantize.py
│   ├── train.py
│   └── deploy/
│       └── convert_to_c.py
└── stm32/
    └── README_STM32.md
```

## Recommended next steps
- Replace the synthetic dataset with your own CSV data.
- If using images, reshape inputs using `--reshape`.
- Use `docs/ARCHITECTURE.md` to understand the model and pipeline flow.
- Follow `stm32/README_STM32.md` for the final STM32 integration.

## Documentation
- `docs/OVERVIEW.md` — big picture and goals
- `docs/ARCHITECTURE.md` — internal design and data flow
- `docs/STM32_DEPLOYMENT_GUIDE.md` — detailed deploy steps
- `docs/CONTRIBUTING.md` — how to contribute to the repo
- `docs/FAQ.md` — beginner-friendly questions and answers
- `docs/PROJECT_CHECKLIST.md` — publishing readiness checklist

## Contribution
This project is open for contributions. Please read `docs/CONTRIBUTING.md` before submitting a pull request.

## License
This repository is licensed under the MIT License. See `LICENSE`.
