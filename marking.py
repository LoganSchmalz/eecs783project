import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import easyocr
import os
import glob


def disp_image(image, title="Image", save_image=False):
    (h, w) = image.shape[:2]
    img = cv2.resize(image, (int(w / 4), int(h / 4)))
    cv2.imshow(title, img)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


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


# contrast function from @/pietz on StackOverflow:
def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)


img_list = [
    "marking_images/A-J-28SOP-03F-SM.png",
    "marking_images/C-T-08DIP-11F-SM.png",
    "marking_images/C-T-48QFP-19F-SM.png",
    "marking_images/C-T-48QFP-20F-SM.png",
]

# img = cv2.imread("marking_images/A-J-28SOP-03F-SM.png")
# img = cv2.imread("marking_images/C-T-08DIP-11F-SM.png")
# img = cv2.imread("marking_images/C-T-48QFP-19F-SM.png")
# img = cv2.imread("marking_images/C-T-48QFP-20F-SM.png")

temp = 1
for i in img_list:
    img = cv2.imread(i)
    # Logan messing with preprocessing starts here
    disp_image(img, "")
    blur2 = cv2.bilateralFilter(img, 9, 150, 150)
    disp_image(blur2, "")
    blur2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)
    disp_image(blur2, "")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = blur2
    # messing with preprocessing ends here
    _, th = threshold(img, cv2.THRESH_BINARY_INV)
    blur = cv2.GaussianBlur(th, (51, 51), 0)
    _, th = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    img = img & th

    # -------- comment out this section to get rid of contrast
    disp_image(
        img, ""
    )  # shows image after threshholding and before adjusting brightness/contrast
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = adjust_contrast_brightness(
        img, 2, 25
    )  # I arbitrarily chose numbers and adjusted from there, feel free to edit
    # ---------- ^^

    os.chdir("th_imgs")  # change directory to th_imgs folder
    cv2.imwrite("thImg_{}.png".format(temp), img)  # save the updated thresholded image
    disp_image(img, "")  # shows image after adjusting brightness/contrast
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    temp = temp + 1
    os.chdir("..")


th_img_list = glob.glob(
    "th_imgs/*.png"
)  # create a list of all images with .png filetype in the th_imgs folder
# print(th_img_list)

reader = easyocr.Reader(["en"])  # OCR reader object

for img in th_img_list:
    result = reader.readtext(img)
    print("\nMarkings on " + img, ": \n")
    for detection in result:
        print(detection[1])  # for all imgs in list, read text with OCR reader and print
