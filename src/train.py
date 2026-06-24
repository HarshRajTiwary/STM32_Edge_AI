import argparse
import os
import numpy as np

from src.data import load_csv
from src.preprocess import fit_transform_save
from src.model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV file path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--model", default="mlp", choices=["mlp", "cnn2d", "ds_cnn", "conv1d", "lstm"], help="Model architecture type")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of output classes; use 1 for regression")
    parser.add_argument("--reshape", default=None, help="comma-separated input shape for CNN/LSTM, e.g. 28,28,1 or 100,1")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    (X_train, y_train), (X_test, y_test) = load_csv(args.data, args.target)
    X_train = fit_transform_save(X_train, out_path=os.path.join(args.out, "scaler.joblib"))
    X_test = np.asarray(X_test)

    if args.reshape:
        shape = tuple(int(x) for x in args.reshape.split(","))
        X_train = X_train.reshape((-1,) + shape)
        X_test = X_test.reshape((-1,) + shape)
    input_shape = X_train.shape[1:]
    model = build_model(args.model, input_shape, num_classes=args.num_classes)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs, batch_size=args.batch)

    # Save Keras model and SavedModel for conversion
    model_path = os.path.join(args.out, "keras_model.h5")
    saved_model_path = os.path.join(args.out, "saved_model")
    model.save(model_path)
    # Export SavedModel format (Keras 3+): use model.export()
    try:
        model.export(saved_model_path)
    except Exception:
        # Fallback for older Keras: use tf.saved_model.save
        import tensorflow as tf

        tf.saved_model.save(model, saved_model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    main()
