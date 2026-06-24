# Project Architecture

This section explains how the repository is structured and how each component works together.

## Overall workflow
1. `data/` stores datasets and generated samples.
2. `src/generate_synthetic.py` creates a synthetic regression dataset for demos.
3. `src/data.py` loads CSV datasets and splits into train/test sets.
4. `src/preprocess.py` scales numeric features and saves a scaler for future inference.
5. `src/model.py` provides multiple STM32-ready model architectures.
6. `src/train.py` trains the selected model and exports it.
7. `src/quantize.py` converts a SavedModel to a quantized `.tflite` model.
8. `src/deploy/convert_to_c.py` converts `.tflite` into a C array for STM32 apps.
9. `stm32/README_STM32.md` explains deployment strategies for TFLM and Cube.AI.

## Key design goals
- **Simplicity:** easy-to-follow scripts and commands
- **Modularity:** separate data, model, training, quantization, and deployment
- **STM32 compatibility:** small models, int8 quantization, TFLM conversion
- **Beginner-friendly:** full explanations and sample commands

## Supported model types
- `mlp`: Multi-layer perceptron, best for tabular data
- `cnn2d`: 2D convolutional network, best for image-like inputs
- `ds_cnn`: depthwise separable CNN, smaller footprint and faster on MCU
- `conv1d`: 1D convolutional network, good for time-series and audio
- `lstm`: sequential model for time-series or sequence inputs

## Deployment-ready artifacts
- `outputs/keras_model.h5` for Keras reference
- `outputs/saved_model` for TensorFlow/TFLite conversion
- `outputs/model_int8.tflite` for embedded inference
- `stm32/model_data.c` for direct inclusion in STM32 projects
