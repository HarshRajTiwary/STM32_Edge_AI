import argparse
import os
import tensorflow as tf
import numpy as np


def load_saved_model(path):
    return tf.saved_model.load(path)


def convert_to_tflite(saved_model_dir, output_tflite, repr_data_gen=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if repr_data_gen is not None:
        converter.representative_dataset = repr_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(output_tflite, "wb") as f:
        f.write(tflite_model)


def representative_from_csv(csv_path, target_col, sample_count=100):
    import pandas as pd

    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col]).values.astype(np.float32)

    def gen():
        for i in range(min(sample_count, X.shape[0])):
            yield [X[i : i + 1]]

    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model", required=True)
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--repr-csv", help="CSV file for representative dataset")
    parser.add_argument("--target", help="Target column in representative CSV")
    args = parser.parse_args()

    gen = None
    if args.repr_csv and args.target:
        gen = representative_from_csv(args.repr_csv, args.target)

    os.makedirs(os.path.dirname(args.tflite) or ".", exist_ok=True)
    convert_to_tflite(args.saved_model, args.tflite, repr_data_gen=gen)
    print("Converted and saved tflite to", args.tflite)


if __name__ == "__main__":
    main()
