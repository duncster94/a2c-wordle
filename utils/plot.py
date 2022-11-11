from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

CELL_SIZE = 0.15
CELL_PADDING = 0.01


def plot_board(state: torch.Tensor):

    fig, ax = plt.subplots(1, 1)

    state_ = state.movedim(0, 2)
    state_ = state_.reshape((state_.shape[0], state_.shape[1], 26, 3))

    # transform `state` into (row, column, letter, color) tuples
    for row_idx, row in enumerate(state_):
        for col_idx, col in enumerate(row):

            # get coords of letter, color
            letter_idx = torch.nonzero(col).flatten()

            # skip empty cell
            if len(letter_idx) == 0:
                continue

            letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[letter_idx[0].item()]
            color = ["#6baa65", "#cab459", "#777c7f", "#ffffff"][letter_idx[1].item()]

            edge_color = "black" if color == "#ffffff" else "white"

            rect = Rectangle(
                (
                    col_idx * CELL_SIZE + CELL_PADDING,
                    -row_idx * CELL_SIZE + CELL_PADDING,
                ),
                CELL_SIZE - 2 * CELL_PADDING,
                CELL_SIZE - 2 * CELL_PADDING,
                facecolor=color,
                edgecolor=edge_color,
            )
            ax.add_patch(rect)

            rect_x, rect_y = rect.get_xy()
            center_x, center_y = (
                rect_x + rect.get_width() / 2,
                rect_y + rect.get_height() / 2,
            )

            text_color = "black" if color == "#ffffff" else "white"

            ax.annotate(
                letter,
                (center_x, center_y),
                color=text_color,
                fontsize=14,
                weight="bold",
                ha="center",
                va="center",
            )

    ax.set_ylim(-5 * CELL_SIZE, CELL_SIZE)
    ax.set_xlim(0, 5 * CELL_SIZE)
    ax.set_aspect(5.5 / 6)
    ax.set_axis_off()
    plt.savefig(
        Path("plots/board") / f"A2C_{str(datetime.now()).replace(':', '-')}.png"
    )

    plt.close(fig)


def plot_values(values: list, name: str):

    fig, ax = plt.subplots(1, 1)

    episodes = list(range(len(values)))

    plt.plot(episodes, values)

    conv_size = 25
    moving_avg = np.convolve(np.array(values), np.ones(conv_size), "valid") / conv_size

    plt.plot(episodes[conv_size - 1 :], moving_avg, c="orange")

    plt.savefig(
        Path("plots/diagnostics")
        / f"{str(datetime.now()).replace(':', '-')}_{name}.png"
    )

    plt.close(fig)
