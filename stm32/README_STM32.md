STM32 Deployment Notes

Options
- TensorFlow Lite for Microcontrollers (TFLM): small runtime, cross-platform. Use when you can compile TFLM for your target MCU.
- STM32Cube.AI: STM32-specific tool to import Keras/TFLite models and generate optimized C code using CMSIS-NN.

TFLM workflow (recommended for lightweight models)
1. Convert model to TFLite with integer quantization (see `src/quantize.py`).
2. Convert `.tflite` to C array using `src/deploy/convert_to_c.py` or `xxd -i`.
3. Add the generated `.c` file to your STM32 project and include initialization code for the interpreter.
4. Build TFLM for your MCU (follow TFLM docs) and link with your project.

STM32Cube.AI workflow
1. Open STM32CubeMX / STM32Cube.AI and import the Keras model or TFLite.
2. Cube.AI will generate optimized C code and integrate with a HAL project.
3. Review resource estimates and tune model/quantization if needed.

Tips
- Start with a very small model and run inference tests on host with the TFLite Interpreter first.
- Check SRAM/Flash estimates in Cube.AI and adjust model size accordingly.
