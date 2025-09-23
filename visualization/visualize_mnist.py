from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
import numpy as np



def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help="")
    args = parser.parse_args()

    plt.ion()
    inputs_np, labels_np = pickle.load(open(args.data, "rb"))

    fig, axs = plt.subplots(1, 3)
    fig.show()
    plt.pause(0.01)

    rand_idxs = np.random.randint(0, inputs_np.shape[0], size=len(axs))
    for ax, idx in zip(axs, rand_idxs, strict=True):
        if len(inputs_np.shape) == 3:
            ax.imshow(inputs_np[idx, :], cmap="gray")
        else:
            ax.imshow(inputs_np[idx, :])
        print(f"Label: {labels_np[idx]}")

    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.show()
    plt.pause(0.01)
    plt.show(block=True)


if __name__ == "__main__":
    main()
