import torchvision
import random
import numpy as np
from tqdm import tqdm


def get_image_and_target(mnist_image, position=None):
    img_as_array = np.asarray(mnist_image)
    canvas = np.zeros((64, 64), dtype=img_as_array.dtype)
    pos_x = random.randint(0, 64 - img_as_array.shape[1] - 1)
    pos_y = random.randint(0, 64 - img_as_array.shape[0] - 1)
    if position is not None:
        pos_y = position[0]
        pos_x = position[1]
    target_x = pos_x + img_as_array.shape[1] // 2
    target_y = pos_y + img_as_array.shape[0] // 2
    random_noise = (np.random.rand(64, 64) * 25).astype(canvas.dtype)
    canvas[pos_y:pos_y + img_as_array.shape[0], pos_x:pos_x + img_as_array.shape[1]] = img_as_array
    canvas = np.clip(canvas.astype(np.float32) + random_noise, 0, 255).astype(img_as_array.dtype)
    return canvas, [target_y, target_x]


def create_data_quadrant_point():
    data_images = []
    data_targets = []

    for i in range(64 - 8):
        for j in range(64 - 8):
            pos = [i, j]
            if pos[0] + 4 > 32 and pos[1] + 4 > 32:
                continue
            canvas, target = get_image_and_target(np.ones((8, 8), dtype=np.uint8) * 255, pos)
            data_images.append(canvas)
            data_targets.append(target)

    np.savez("data/not_so_clvr_train.npz", images=np.array(data_images), targets=np.array(data_targets))

    data_images = []
    data_targets = []

    for i in range(64 - 8):
        for j in range(64 - 8):
            pos = [i, j]
            if not(pos[0] + 4 > 32 and pos[1] + 4 > 32):
                continue
            canvas, target = get_image_and_target(np.ones((8, 8), dtype=np.uint8) * 255, pos)
            data_images.append(canvas)
            data_targets.append(target)

    np.savez("data/not_so_clvr_val.npz", images=np.array(data_images), targets=np.array(data_targets))


def create_data_quadrant():
    mnist = torchvision.datasets.MNIST("data/download", download=True, train=True)

    data_images = []
    data_targets = []
    indices = list(range(len(mnist)))

    i = 0
    dim = 64 - 28
    for index in indices[:50000]:
        i = (i + 1) % (dim * dim)
        pos = [i // dim, i % dim]
        while pos[0] + 14 > 32 and pos[1] + 14 > 32:
            i = (i + 1) % (dim * dim)
            pos = [i // dim, i % dim]
        canvas, target = get_image_and_target(mnist[index][0], pos)
        data_images.append(canvas)
        data_targets.append(target)

    np.savez("data/floating_mnist_quadrant_train.npz", images=np.array(data_images), targets=np.array(data_targets))

    data_images = []
    data_targets = []

    i = 0
    dim = 64 - 28
    for index in indices[50000:]:
        i = (i + 1) % (dim * dim)
        pos = [i // dim, i % dim]
        while not (pos[0] + 14 > 32 and pos[1] + 14 > 32):
            i = (i + 1) % (dim * dim)
            pos = [i // dim, i % dim]
        canvas, target = get_image_and_target(mnist[index][0], pos)
        data_images.append(canvas)
        data_targets.append(target)

    np.savez("data/floating_mnist_quadrant_val.npz", images=np.array(data_images), targets=np.array(data_targets))


def create_data_uniform():
    mnist = torchvision.datasets.MNIST("data/download", download=True, train=True)
    data_images = []
    data_targets = []

    indices = list(range(len(mnist)))
    random.shuffle(indices)

    for i in tqdm(indices[:50000]):
        canvas, target = get_image_and_target(mnist[i][0])
        data_images.append(canvas)
        data_targets.append(target)

    np.savez("data/floating_mnist_uniform_train.npz", images=np.array(data_images), targets=np.array(data_targets))

    data_images = []
    data_targets = []

    for i in tqdm(indices[50000:]):
        canvas, target = get_image_and_target(mnist[i][0])
        data_images.append(canvas)
        data_targets.append(target)

    np.savez("data/floating_mnist_uniform_val.npz", images=np.array(data_images), targets=np.array(data_targets))

    generate_test = False

    if generate_test:
        mnist = torchvision.datasets.MNIST("data/download", download=True, train=False)
        data_images = []
        data_targets = []
        indices = list(range(len(mnist)))

        for i in tqdm(indices):
            canvas, target = get_image_and_target(mnist[i][0])
            data_images.append(canvas)
            data_targets.append(target)

        np.savez("data/floating_mnist_uniform_test.npz", images=np.array(data_images), targets=np.array(data_targets))


if __name__ == '__main__':
    create_data_quadrant_point()
