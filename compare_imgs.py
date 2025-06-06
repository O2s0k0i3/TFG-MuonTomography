import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compare_all(output_path):
    file = open(output_path, "w")
    file.write("img_num,true_positives/true,false_positive/false\n")
    for img_num in range(100):
        # skip the images that aren't recognizable
        if img_num in [2, 3, 8, 9, 16, 22, 24, 25, 29, 30, 34, 38, 44, 48, 54, 59, 63, 68, 74, 77, 81, 88, 89, 95, 97, 98]:
            continue
        result = compare_imgs(sim_path(img_num), trained_path(img_num), trained_threshold=180)
        file.write(f"{img_num},{result[0]},{result[1]}\n")
    file.close()

def show_imgs(sim_img, trained_img, sim, output):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(np.array(sim_img), cmap='binary')
    axes[0, 0].set_title("Simulated")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.array(trained_img), cmap='binary')
    axes[0, 1].set_title("Trained")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(np.array(sim), cmap='binary')
    axes[1, 0].set_title("Binary Simulated")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(output, cmap='binary')
    axes[1, 1].set_title("Binary Trained")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def compare_imgs(sim_path, trained_path, trained_threshold = 150, show = False):
    BLACK = 0
    WHITE  = 255

    sim_img = Image.open(sim_path).convert("L")
    trained_img = Image.open(trained_path).convert("L")

    sim = np.array(sim_img)

    sim_threshold = sim.min()

    sim = np.array(sim_img.point(lambda p: WHITE if p > sim_threshold else BLACK))
    output = np.array(trained_img.point(lambda p: WHITE if p > trained_threshold else BLACK))

    if show:
        show_imgs(sim_img, trained_img, sim, output)

    true_positives = 0
    blacks = 0
    false_positives = 0
    whites = 0

    for i in range(512):
        for (sim_px, trained_px) in zip(sim[i], output[i]):
            if sim_px == BLACK:
                blacks += 1
                if trained_px == BLACK:
                    true_positives += 1
                continue
            whites += 1
            if trained_px == BLACK:
                false_positives += 1

    return (true_positives/blacks, false_positives/whites)


sim_path = lambda i: f"../trained_images/evaluated/output_{i}/sim_{i}.png"
trained_path = lambda i: f"../trained_images/evaluated/output_{i}/output_{i}.png"
output_path = "../trained_images/evaluated/other.csv"

# print(compare_imgs(sim_path(73), trained_path(73), show=True, trained_threshold=190))

# compare_all(output_path)

file = np.loadtxt("../trained_images/evaluated/other.csv", skiprows=1, delimiter=",")

true_pos = []
false_pos = []

for row in file:
    true_pos.append(row[1] * 100)
    false_pos.append(row[2] * 100)

# plt.hist(true_pos)
# plt.xlim([0, 100])
# plt.title("True positive")
# plt.savefig("../trained_images/evaluated/mala-true_positives.svg", format="svg")
# plt.show()

# plt.hist(false_pos)
# plt.xlim([0, 100])
# plt.title("False positive")
# plt.savefig("../trained_images/evaluated/mala-false_positives-0-100.svg", format="svg")
# plt.show()

plt.hist(false_pos)
plt.xlim([0, 40])
plt.title("False positive")
# plt.savefig("../trained_images/evaluated/mala-false_positives-0-15.svg", format="svg")
plt.show()
