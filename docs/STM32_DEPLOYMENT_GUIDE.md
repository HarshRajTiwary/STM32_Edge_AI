# STM32 Deployment Guide

This guide explains how to deploy the generated TFLite model to an STM32 microcontroller.

## Recommended STM32 deployment paths

### 1. TensorFlow Lite for Microcontrollers (TFLM)
- Best for tiny models and direct control over code.
- Use `.tflite` files and model data arrays.
- Supported on many STM32 boards with CMSIS-NN.

### 2. STM32Cube.AI
- Best for automatic C code generation and performance tuning.
- Import Keras or TFLite models into STM32Cube.AI.
- Generates optimized code for STM32 hardware.

## Steps for TFLM deployment
1. Generate and quantize the model.
2. Convert the `.tflite` file to a C array:
   ```bash
   .venv/bin/python src/deploy/convert_to_c.py outputs/model_int8.tflite stm32/model_data.c model_data
   ```
3. Add `stm32/model_data.c` into your STM32CubeIDE or Makefile project.
4. Include the array in your application and create a TFLM interpreter.
5. Feed quantized input data and retrieve int8 output.

## Notes for STM32Cube.AI
- Open STM32Cube.AI and import the model.
- Use Keras `keras_model.h5` or the `.tflite` file.
- Cube.AI generates C code and estimates memory usage.
- Validate model size before flashing to the MCU.

## Best practices
- Keep models small and quantized to int8.
- Validate the TFLite model on the desktop before deploying.
- Test with representative input shapes.
- Use the smallest board that still fits the model.
