import pickle
import numpy as np
from utils.mnist_utils import colors
from argparse import ArgumentParser


def grayscale_to_color(img, color):
    """Convert a 28x28 grayscale image to color by tinting with the given RGB color."""
    rgb_img = np.zeros((28, 28, 3))
    for c in range(3):
        rgb_img[:, :, c] = (img * color[c])
    rgb_img = rgb_img / 255.0
    return rgb_img


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--data_in', type=str, required=True, help="")
    parser.add_argument('--data_out', type=str, required=True, help="")
    args = parser.parse_args()

    inputs_np, labels_np = pickle.load(open(args.data_in, "rb"))

    inputs_np_color = np.zeros(inputs_np.shape + (3,))
    labels_np_color = np.zeros(labels_np.shape + (2,))
    for idx in range(inputs_np.shape[0]):
        img = inputs_np[idx]

        # Randomly pick a color
        color_idx = np.random.randint(len(colors))
        _, color_val = colors[color_idx]

        # Convert to colored image
        inputs_np_color[idx] = grayscale_to_color(img, color_val)
        labels_np_color[idx] = np.array([labels_np[idx], color_idx])

    pickle.dump((inputs_np_color, labels_np_color), open(args.data_out, "wb"), protocol=-1)

if __name__ == "__main__":
    main()
