# FAQ

## What is this project for?
This project shows how to train machine learning models on a PC and deploy them to STM32 microcontrollers using TensorFlow Lite.

## Which models can I deploy?
- MLP (`mlp`) for tabular data
- 2D CNN (`cnn2d`) for image-like inputs
- Depthwise separable CNN (`ds_cnn`) for smaller footprint
- 1D CNN (`conv1d`) for time-series and audio
- LSTM (`lstm`) for sequence data

## How do I choose a model?
- Use `mlp` for structured numeric datasets.
- Use `cnn2d` or `ds_cnn` for image or 2D sensor data.
- Use `conv1d` for audio, vibration, or sequential sensor readings.
- Use `lstm` for time-series where order matters.

## Do I need a GPU?
No. CPU-only training works fine for this demo pipeline.

## Where do I put my dataset?
Place CSV files in the `data/` folder.

## How do I test the TFLite model?
Convert to TFLite and run a desktop TFLite interpreter, or use the generated C array in a microcontroller project.

## What if I want a different model type?
Add it in `src/model.py` and update the README examples.
