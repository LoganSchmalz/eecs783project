import cv2
import numpy as np


def disp_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def remove_shadows(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to the LAB color space to work on the lightness channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back the LAB channels
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    # disp_image(img_enhanced, "Enhanced Image")

    # Convert to grayscale to detect shadows
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)

    sigma = 100
    blur2 = cv2.bilateralFilter(gray, 7, sigma, sigma)
    # disp_image(blur2, "Bilateral filtered Image")

    # adaptive threshold 11, 3
    img1 = cv2.adaptiveThreshold(
        blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    )
    disp_image(img1, "Adaptive Thresholding 1")

    # # blur again
    # blur3 = cv2.medianBlur(img1, 5)
    # disp_image(blur3, "Median Blurred Image")

    # # adaptive threshold 11, 3
    # img2 = cv2.adaptiveThreshold(
    #     blur3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    # )
    # disp_image(img2, "Adaptive Thresholding 2")

    # Apply binary thresholding to segment shadow regions
    return img1


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


# Example usage
image_corpus = [
    "./pins_images/A-J-28SOP-01B-SM.png",
    "./pins_images/A-D-64QFP-14B-SM.png",
    "./pins_images/A-D-64QFP-15B-SM.png",
    "./pins_images/C-T-28SOP-04F-SM.png",
]
input_image = image_corpus[3]  # Path to the input image
shadowless_img = remove_shadows(input_image)


img = cv2.imread(input_image)
img_rgb = cv2.imread(input_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, th = threshold(img, cv2.THRESH_BINARY_INV)
# blur = cv2.GaussianBlur(th, (51, 51), 0)
# _, th = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)
# th = cv2.convertScaleAbs(th, alpha=1, beta=0).astype(np.int32)
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # draw in blue the contours that were founded
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    cv2.rectangle(shadowless_img, (x, y), (x + w, y + h), (255, 255, 255), -1)

disp_image(shadowless_img, "Rect Image")

edges = cv2.Canny(shadowless_img, 20, 100)
disp_image(edges, "Canny")

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(f"Number of contours: {len(contours)}")

bounding_boxes = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    if 100 < area < 15000 and (10 < w) and (10 < h):
        # if (5 < w < 200) and (5 < h < 200):
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

print(len(bounding_boxes))
disp_image(img_rgb, "Bounding Boxes")

# disp_image(cv2.merge([img, shadowless_img]), "Final Image")
# disp_image(img, "Shadowless Image")


cv2.destroyAllWindows()
