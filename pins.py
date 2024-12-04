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


# img = cv2.imread("pins_images/A-D-64QFP-14B-SM.png")
img = cv2.imread("pins_images/A-D-64QFP-15B-SM.png")
# img = cv2.imread("pins_images/A-J-28SOP-01B-SM.png")
# img = cv2.imread("pins_images/C-T-28SOP-04F-SM.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(h, w) = img.shape[:2]
img = cv2.resize(img, (int(w / 4), int(h / 4)))

_, th = threshold(img, cv2.THRESH_BINARY_INV)
blur = cv2.GaussianBlur(th, (51, 51), 0)
_, th = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)
# th = cv2.convertScaleAbs(th, alpha=1, beta=0).astype(np.int32)
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
th = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

if len(contours) != 0:
    # draw in blue the contours that were founded
    # cv2.drawContours(th, contours, -1, 255, 3)

    # find the biggest countour (c) by the area
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    # cv2.rectangle(th, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 10)
cv2.imshow("", img)
cv2.waitKey(0)

# _, th = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
# edges = cv2.Canny(img | th, 100, 200)
# cv2.imshow("res2", edges)
# cv2.waitKey(0)
# _, th = cv2.threshold(img, 250, 0, cv2.THRESH_BINARY)
cv2.imshow("", th)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(len(contours))
print(contours)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
cv2.drawContours(img, contours, -1, 255, 3)
cv2.imshow("res2", img)
cv2.waitKey(0)

# cv2.imshow("res2", th)
# cv2.waitKey(0)
# cv2.imshow("res2", np.bitwise_or(img | th, edges[:, :]))
# cv2.waitKey(0)
cv2.destroyAllWindows()


# Z = img.reshape((-1, 3))
# # convert to np.float32
# Z = np.float32(Z)
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 4
# ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
# cv2.imshow("res2", res2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
