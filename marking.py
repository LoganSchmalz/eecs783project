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
    # cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def isolate_center(orig_img: np.ndarray) -> (int, int, int, int):
    my_img = orig_img.copy()

    my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    # apply threshold to mask IC housing
    blur2 = cv2.bilateralFilter(my_img, 9, 75, 75)
    _, th = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were founded
        print("Len Contours", len(contours))
        # sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        c = max(contours, key=cv2.contourArea)
        # cv2.drawContours(th2, sorted_contours[:1], -1, (255, 255, 255), -1)
        x, y, w, h = cv2.boundingRect(c)
        return (x + 100, y + 100, w - 100, h - 100)

    return (0, 0, my_img.shape[0], my_img.shape[1])


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

    # -------- comment out this section to get rid of contrast
    disp_image(
        img, "original"
    )  # shows image after threshholding and before adjusting brightness/contrast
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    img = adjust_contrast_brightness(
        img, 2, 25
    )  # I arbitrarily chose numbers and adjusted from there, feel free to edit
    # ---------- ^^

    center = isolate_center(img)
    img = img[
        center[1] : center[1] + center[3] - 100, center[0] : center[0] + center[2] - 100
    ]
    # maybe adjust 30 to different values here
    img = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)
    disp_image(img, "contrast + denoise")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    disp_image(img, "grayscale")
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    disp_image(th, "threshold")
    img = th

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
