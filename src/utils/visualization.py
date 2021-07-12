import matplotlib.pyplot as plt
from skimage import color
import numpy as np
from typing import List
import cv2


def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane, cmap=cmap)
    ax.axis("off")

    if title:
        ax.set_title(title)


def explore_slices_multichannel(data, cmap="gray"):
    from ipywidgets import interact

    N = len(data)

    @interact(plane=(0, N - 1), channel=(0, data.shape[1] - 1))
    def display_slice(plane=8, channel=0):
        fig, ax = plt.subplots(figsize=(10, 8))

        show_plane(
            ax,
            data[plane, channel],
            title="Plane {}, channel {}".format(plane, channel),
            cmap=cmap,
        )

        plt.show()

    return display_slice


def explore_slices(data, cmap="gray"):
    from ipywidgets import interact

    N = len(data)

    @interact(plane=(0, N - 1))
    def display_slice(plane=8):
        fig, ax = plt.subplots(figsize=(10, 8))

        show_plane(ax, data[plane], title="Plane {}".format(plane), cmap=cmap)

        plt.show()

    return display_slice


def explore_slices_2_samples(data, cmap="gray"):
    from ipywidgets import interact

    N = min(len(data[0]), len(data[1]))

    @interact(plane=(0, N - 1))
    def display_slice(plane=8):
        fig, ax = plt.subplots(figsize=(20, 10), nrows=1, ncols=2)

        show_plane(ax[0], data[0][plane], title="Plane {}".format(plane), cmap=cmap)
        show_plane(ax[1], data[1][plane], title="Plane {}".format(plane), cmap=cmap)
        plt.show()

    return display_slice


def explore_slices_2_samples_multichannel(data, cmap="gray"):
    from ipywidgets import interact

    N = min(len(data[0]), len(data[1]))
    n_channels = min(data[0].shape[1], data[1].shape[1])

    @interact(plane=(0, N - 1), channel=(0, n_channels - 1))
    def display_slice(plane=8, channel=0):
        fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=2)

        show_plane(
            ax[0],
            data[0][plane, channel],
            title="Plane {}, channel {}".format(plane, channel),
            cmap=cmap,
        )
        show_plane(
            ax[1],
            data[1][plane, channel],
            title="Plane {}, channel {}".format(plane, channel),
            cmap=cmap,
        )
        plt.show()

    return display_slice


def color_3d_segmentation(mask: np.ndarray, intensity_image: np.ndarray):
    mask = mask.astype(int)
    if intensity_image.max() > 255:
        intensity_image = np.uint8((intensity_image / intensity_image.max()) * 255)
    colored_segmentation = []
    for j in range(len(mask)):
        colored_segmentation.append(
            color.label2rgb(mask[j], intensity_image[j], alpha=0.2, bg_label=0)
        )
    return np.array(colored_segmentation)


def color_3d_segmentations(masks, intensity_images):
    colored_segmentations = []
    for i in range(len(masks)):
        mask = masks[i]
        mask = mask.astype(int)
        intensity_image = intensity_images[i]
        colored_segmentations.append(
            color_3d_segmentation(mask=mask, intensity_image=intensity_image)
        )
    return colored_segmentations


def plot_colored_3d_segmentation(mask, intensity_image):
    mask = mask.astype(int)
    if intensity_image.max() > 255:
        intensity_image = np.uint8((intensity_image / intensity_image.max()) * 255)
    colored_segmentation = color_3d_segmentation(
        mask=mask, intensity_image=intensity_image
    )
    start = 0
    end = len(mask)
    for i in range(len(mask)):
        if mask[i].any():
            start = i - 1
            break
    for i in range(start + 1, len(mask)):
        if not mask[i].any():
            end = i + 1
            break
    depth = end - start
    fig, ax = plt.subplots(
        nrows=3,
        ncols=depth,
        figsize=[depth * 3, 9],
        gridspec_kw={"wspace": 0.0, "hspace": 0.1},
    )
    for j in range(start, end):
        ax[0, j - start].imshow(
            cv2.resize(intensity_image[j], dsize=(64, 64)), vmin=0, vmax=255
        )
        ax[1, j - start].imshow(
            cv2.resize(mask[j].astype(float), dsize=(64, 64)),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        ax[2, j - start].imshow(
            cv2.resize(colored_segmentation[j], dsize=(64, 64)), vmin=0, vmax=255
        )

        ax[0, j - start].axis("off")
        ax[1, j - start].axis("off")
        ax[2, j - start].axis("off")
        ax[0, j - start].set_aspect("auto")
        ax[1, j - start].set_aspect("auto")
        ax[2, j - start].set_aspect("auto")
    fig.subplots_adjust(hspace=0.1, wspace=0.0)
    return fig, ax


def plot_3d_images_as_map(
    images: List[np.ndarray], n_images_per_slide: int = 5, max_depth: int = 23,
):
    depth = max_depth
    figures = []
    fig = None
    for i in range(len(images)):
        idx = i % n_images_per_slide
        if idx == 0:
            fig, ax = plt.subplots(
                nrows=min(n_images_per_slide, len(images) - i),
                ncols=depth,
                figsize=[40.0, 10.0],
                gridspec_kw={"wspace": 0.0, "hspace": 0.0},
            )

        for j in range(depth):
            if j < np.squeeze(images[i]).shape[0]:
                img = np.squeeze(images[i])
                img = img[j]
                img = cv2.resize(img, dsize=(64, 64))
                # if img.max() > 0 :
                #    img = img / img.max()
            else:
                img = np.zeros([64, 64])
            ax[idx, j].imshow(
                img, interpolation="nearest", cmap="magma",
            )
            ax[idx, j].axis("off")
            ax[idx, j].set_aspect("auto")
        if idx == n_images_per_slide - 1:
            figures.append(fig)
            fig.subplots_adjust(hspace=0.0, wspace=0.0)
            plt.subplots_adjust(hspace=0.0, wspace=0.0)
            plt.show()
