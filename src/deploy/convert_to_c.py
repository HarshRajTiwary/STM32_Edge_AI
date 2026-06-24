import sys
import os


def tflite_to_c(tflite_path, c_out_path, array_name="model_data"):
    with open(tflite_path, "rb") as f:
        data = f.read()

    lines = []
    lines.append("#include <stdint.h>")
    lines.append(f"const unsigned char {array_name}[] = {{")
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        line = ", ".join(str(b) for b in chunk)
        lines.append("    " + line + ",")
    lines.append("};")
    lines.append(f"const unsigned int {array_name}_len = {len(data)};")

    os.makedirs(os.path.dirname(c_out_path) or ".", exist_ok=True)
    with open(c_out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    if len(sys.argv) < 3:
        print("Usage: convert_to_c.py model.tflite out.c [array_name]")
        sys.exit(1)
    tflite_path = sys.argv[1]
    c_out = sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "model_data"
    tflite_to_c(tflite_path, c_out, array_name=name)
    print("Wrote C file:", c_out)


if __name__ == "__main__":
    main()
