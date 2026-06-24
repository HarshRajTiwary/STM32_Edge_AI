# STM32 ML Deployment Pipeline

This repository provides an end-to-end pipeline to train models locally and deploy them to STM32 microcontrollers.

Setup
- Create a Python 3.8+ virtual environment and install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Workflow overview
1. Add or copy your dataset into the `data/` folder. For tabular data use CSV files.
2. Train a model with `src/train.py`.
3. Quantize the model to a `.tflite` using `src/quantize.py` (use a representative dataset CSV for best int8 accuracy).
4. Convert the `.tflite` to a C array via `src/deploy/convert_to_c.py`.
5. Follow `stm32/README_STM32.md` for integrating with your chosen STM32 workflow (TFLM or STM32Cube.AI).

Example commands

```
python src/train.py --data data/mydata.csv --target label --epochs 20 --out outputs
python src/train.py --data data/mydata.csv --target label --model cnn2d --reshape 28,28,1 --epochs 20 --out outputs
python src/train.py --data data/mydata.csv --target label --model ds_cnn --reshape 28,28,1 --epochs 20 --out outputs
python src/train.py --data data/mydata.csv --target label --model conv1d --reshape 100,1 --epochs 20 --out outputs
python src/train.py --data data/mydata.csv --target label --model lstm --reshape 100,1 --epochs 20 --out outputs
python src/quantize.py --saved_model outputs/saved_model --tflite outputs/model_int8.tflite --repr-csv data/mydata_sample.csv --target label
python src/deploy/convert_to_c.py outputs/model_int8.tflite stm32/model_data.c model_data
```

Supported `--model` values:
- `mlp`: dense multi-layer perceptron for tabular or flattened inputs
- `cnn2d`: small 2D convolutional network for image-like inputs
- `ds_cnn`: depthwise separable CNN for smaller footprint on STM32
- `conv1d`: 1D convolutional network for time-series or sequence inputs
- `lstm`: small LSTM recurrent model for sequential data

For CNN/LSTM/Conv1D models, also provide `--reshape` with a comma-separated input shape.

Folder summary
- `data/` - datasets
- `src/` - scripts for data, preprocessing, model, train, quantize, deploy helpers
- `stm32/` - STM32-specific deployment notes
- `outputs/` - artifacts produced by training/quantization

If you want, provide a sample dataset and I will run through a full example and produce the final `.c` file ready to add to an STM32 project.
