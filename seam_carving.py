import matplotlib.pyplot as plt
import numpy as np


def pixel_power(img, y, x):
    dim = np.shape(img)
    y_dim = dim[0]
    x_dim = dim[1]

    power = 0
    for i in range(3):
        grad_x = img[y, (x+1) % x_dim, i] - img[y, x-1, i]
        grad_y = img[(y+1) % y_dim, x, i] - img[y-1, x, i]
        power += grad_x ** 2 + grad_y ** 2

    return power


def pixel_power_tab(img):
    dim = np.shape(img)
    y = dim[0]
    x = dim[1]

    power_tab = np.zeros((y, x))

    for i in range(y):
        for j in range(x):
            power_tab[i, j] = pixel_power(img, i, j)

    return power_tab


def find_vertical_seam(img):
    power_tab = pixel_power_tab(img)
    dim = np.shape(power_tab)
    y = dim[0]
    x = dim[1]

    cost_tab = -np.ones((y, x))
    cost_tab[0, :] = np.copy(power_tab[0, :])
    path_tab = -np.ones((y, x))
    path_tab[0, :] = np.arange(0, x)
    for i in range(1, y):
        for j in range(x):
            up_neighbours_x = np.array(range(max(j - 1, 0), min(j + 1, x - 1)+1))
            min_cost_x = up_neighbours_x[np.argmin(cost_tab[i - 1, up_neighbours_x])]
            cost_tab[i, j] = power_tab[i, j] + cost_tab[i - 1, min_cost_x]
            path_tab[i, j] = min_cost_x

    seam = np.zeros(y, dtype=int)
    seam[y - 1] = np.argmin(cost_tab[y - 1, :])
    for i in range(y - 2, -1, -1):
        seam[i] = path_tab[i, seam[i + 1]]

    return seam


def remove_vertical_seam(img, seam):
    dim = np.shape(img)
    y = dim[0]
    x = dim[1] - 1

    new_img = np.zeros((y, x, 3))
    for i in range(y):
        new_img[i, 0:seam[i], :] = np.copy(img[i, 0:seam[i], :])
        new_img[i, seam[i]:, :] = np.copy(img[i, seam[i]+1:, :])

    return new_img


def vertical_seam_carving(img, num_removed_seams):
    seam = find_vertical_seam(img)
    new_img = remove_vertical_seam(img, seam)
    for i in range(num_removed_seams-1):
        seam = find_vertical_seam(new_img)
        new_img = remove_vertical_seam(new_img, seam)

    return new_img


if __name__ == "__main__":
    img = plt.imread("tower.png")
    new_img = vertical_seam_carving(img, 100)
    plt.imshow(new_img)
    plt.show()
