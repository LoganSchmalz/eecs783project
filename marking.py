import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt


def threshold(img, threshold_setting):
    images = []
    titles = []

    # global thresholding
    global_threshold_val = 127
    ret1, th1 = cv2.threshold(img, global_threshold_val, 255, threshold_setting)
    images.extend(
        [
            img,
            0,
            th1,
        ]
    )
    titles.extend(
        [
            "Original Noisy Image",
            "Histogram",
            f"Global Thresholding (v={global_threshold_val})",
        ]
    )

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, threshold_setting + cv2.THRESH_OTSU)
    images.extend(
        [
            img,
            0,
            th2,
        ]
    )
    titles.extend(
        [
            "Original Noisy Image",
            "Histogram",
            "Otsu's Thresholding",
        ]
    )

    # Otsu's thresholding after Gaussian filtering
    kernel_size = 3
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, threshold_setting + cv2.THRESH_OTSU)
    images.extend(
        [
            blur,
            0,
            th3,
        ]
    )
    titles.extend(
        [
            "Gaussian filtered Image",
            "Histogram",
            "Otsu's Thresholding",
        ]
    )

    # Otsu's thresholding after bilateral filtering
    sigma = 75  # 75
    blur2 = cv2.bilateralFilter(img, 9, sigma, sigma)
    ret4, th4 = cv2.threshold(blur2, 0, 255, threshold_setting + cv2.THRESH_OTSU)
    images.extend(
        [
            blur2,
            0,
            th4,
        ]
    )
    titles.extend(
        [
            "Bilateral filtered Image",
            "Histogram",
            "Otsu's Thresholding",
        ]
    )

    # th5 = cv2.adaptiveThreshold(
    #     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_setting, 11, 2
    # )
    # images.extend(
    #     [
    #         img,
    #         0,
    #         th5,
    #     ]
    # )
    # titles.extend(
    #     [
    #         "Original Noisy Image",
    #         "Histogram",
    #         "Adaptive Thresholding",
    #     ]
    # )

    th6 = cv2.adaptiveThreshold(
        blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_setting, 11, 2
    )
    images.extend(
        [
            blur2,
            0,
            th6,
        ]
    )
    titles.extend(
        [
            "Bilateral filtered Image",
            "Histogram",
            "Adaptive Thresholding",
        ]
    )

    # plot all the images and their histograms
    # i_max = int(len(images) / 3)
    # for i in range(i_max):
    #     plt.subplot(i_max, 3, i * 3 + 1), plt.imshow(images[i * 3], "gray")
    #     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(i_max, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    #     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(i_max, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], "gray")
    #     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    # plt.show()
    # plt.cla()

    return (ret4, th4)


# img = cv2.imread("A-D-64QFP-14B-SM.png")
img = cv2.imread("A-D-64QFP-15B-SM.png")
# img = cv2.imread("A-J-28SOP-01B-SM.png")
# img = cv2.imread("C-T-28SOP-04F-SM.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(h, w) = img.shape[:2]
img = cv2.resize(img, (int(w / 4), int(h / 4)))

_, th = threshold(img, cv2.THRESH_BINARY_INV)
blur = cv2.GaussianBlur(th, (51, 51), 0)
_, th = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
img = img & th
cv2.imshow("", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
