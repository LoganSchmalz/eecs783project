import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
import easyocr
import os
import glob
# import pytesseract


def disp_image(image, title="Image", save_image=False):
    (h, w) = image.shape[:2]
    img = cv2.resize(image, (int(w / 2), int(h / 2)))
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


def kmeans(img, K):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


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
    center = isolate_center(img)
    img = img[
        center[1] + 100 : center[1] + center[3] - 200,
        center[0] : center[0] + center[2] - 175,
    ]
    # (h, w) = img.shape[:2]
    # scale = 2
    # img = cv2.resize(img, (int(w / scale), int(h / scale)))
    disp_image(img, "center")
    # # -------- comment out this section to get rid of contrast
    # img = adjust_contrast_brightness(
    #     img, 2.0, 50
    # )  # I arbitrarily chose numbers and adjusted from there, feel free to edit
    # disp_image(img, "contrast")
    # # ---------- ^^

    # maybe adjust 30 to different values here
    img = cv2.fastNlMeansDenoisingColored(img, None, 9, 9, 7, 21)
    # img = cv2.bilateralFilter(img, 9, 75, 75)
    # img = cv2.GaussianBlur(img, (9, 9), cv2.BORDER_REPLICATE)
    disp_image(img, "denoise")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img, 235, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    disp_image(th, "threshold")
    img = th

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
    disp_image(img, "opening")

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < 200 and (w < 20 or h < 20):
            x, y, w, h = cv2.boundingRect(c)
            # cv2.drawContours(img, c, -1, (0, 255, 0), thickness=cv2.FILLED)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), -1)
    disp_image(img, "contours")
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < 200 and (w < 20 or h < 20):
            x, y, w, h = cv2.boundingRect(c)
            # cv2.drawContours(img, c, -1, (0, 255, 0), thickness=cv2.FILLED)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # kernel_size = 5
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    # disp_image(img, "gradient")
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)
    disp_image(img, "closing")
    # kernel_size = 3
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # erosion = cv2.erode(img, kernel, iterations=1)
    # disp_image(erosion, "erosion")
    # img = erosion
    # # kernel_size = 3
    # # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # # dilation = cv2.dilate(erosion, kernel, iterations=1)
    # # disp_image(dilation, "dilation")
    # # img = dilation

    os.chdir("th_imgs")  # change directory to th_imgs folder
    cv2.imwrite("thImg_{}.png".format(temp), img)  # save the updated thresholded image
    disp_image(img, "")  # shows image after adjusting brightness/contrast
    temp = temp + 1
    os.chdir("..")


th_img_list = glob.glob(
    "th_imgs/*.png"
)  # create a list of all images with .png filetype in the th_imgs folder
print(th_img_list)


# ----- EASY OCR -----
reader = easyocr.Reader(["en"])  # OCR reader object

for img in th_img_list:
    result = reader.readtext(img, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
    print("\nMarkings on " + img, ": \n")
    for detection in result:
        print(detection[1])  # for all imgs in list, read text with OCR reader and print


# ----- TESSERACT -----
# pytesseract.pytesseract.tesseract_cmd = r'/Users/aidan/Library/Python/3.9/bin'
# Perform OCR on an image
# for img in th_img_list:
#     text = pytesseract.image_to_string('img')
#     print("\nMarkings on " + img, ": \n")
#     for line in text:
#         print(text)


# ----- KERAS (DOESN'T WORK) -----
# pipeline = keras_ocr.pipeline.Pipeline()


# ----- OCRA (doesn't work) -----
# you'll just have to:
# >> pip3 install "paddleocr>=2.0.1"
# >> pip3 install paddleocr --upgrade (maybe)
# >> pip3 install paddlepaddle

# ocr = PaddleOCR(use_angle_cls=True, lang='en') # OCRA reader

# for img in th_img_list:
#     result = ocr.ocr(img, cls=True)
#     print("\nMarkings on " + img, ": \n")
#     for line in result:
#         print(line)
