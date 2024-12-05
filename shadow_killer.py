import cv2
import numpy as np


# merging boxes algorithm from https://stackoverflow.com/questions/66490374/
# tuplify
def tup(point):
    return (point[0], point[1])


# returns true if the two boxes overlap
def overlap(source, target):
    # unpack points
    tl1, br1 = source
    tl2, br2 = target
    # checks
    if tl1[0] >= br2[0] or tl2[0] >= br1[0]:
        return False
    if tl1[1] >= br2[1] or tl2[1] >= br1[1]:
        return False
    return True


# returns all overlapping boxes
def getAllOverlaps(boxes, bounds, index):
    overlaps = []
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a)
    return overlaps


def merge(boxes, merge_margin, img=None):
    # this is gonna take a long time
    finished = False
    highlight = [[0, 0], [1, 1]]
    points = [[[0, 0]]]
    while not finished:
        # set end con
        finished = True
        # check progress
        # print("Len Boxes: " + str(len(boxes)))
        # draw boxes # comment this section out to run faster
        if img is not None:
            copy = np.copy(img)
            for box in boxes:
                cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0, 200, 0), 1)
            cv2.rectangle(copy, tup(highlight[0]), tup(highlight[1]), (0, 0, 255), 2)
            for point in points:
                point = point[0]
                cv2.circle(copy, tup(point), 4, (255, 0, 0), -1)
            cv2.imshow("Copy", copy)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

        # loop through boxes
        index = len(boxes) - 1
        while index >= 0:
            # grab current box
            curr = boxes[index]
            # add margin
            tl = curr[0][:]
            br = curr[1][:]
            tl[0] -= merge_margin
            tl[1] -= merge_margin
            br[0] += merge_margin
            br[1] += merge_margin
            # get matching boxes
            overlaps = getAllOverlaps(boxes, [tl, br], index)
            # check if empty
            if len(overlaps) > 0:
                # combine boxes
                # convert to a contour
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    tl, br = boxes[ind]
                    con.append([tl])
                    con.append([br])
                con = np.array(con)
                # get bounding rect
                x, y, w, h = cv2.boundingRect(con)
                # stop growing
                w -= 1
                h -= 1
                merged = [[x, y], [x + w, y + h]]
                # highlights
                highlight = merged[:]
                points = con
                # remove boxes from list
                overlaps.sort(reverse=True)
                for ind in overlaps:
                    del boxes[ind]
                boxes.append(merged)
                # set flag
                finished = False
                break

            # increment
            index -= 1
    cv2.destroyAllWindows()

    return boxes


#####


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
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(50, 50))
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

# apply threshold to mask IC housing
blur2 = cv2.bilateralFilter(img, 9, 75, 75)
_, th = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# blur = cv2.GaussianBlur(th, (51, 51), 0)
# _, th = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)
# th = cv2.convertScaleAbs(th, alpha=1, beta=0).astype(np.int32)
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # draw in blue the contours that were founded
    print("Len Contours", len(contours))
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # c = max(contours, key=cv2.contourArea)
    cv2.drawContours(shadowless_img, sorted_contours[:1], -1, (255, 255, 255), -1)
    # x, y, w, h = cv2.boundingRect(c)

    # # draw the biggest contour (c) in green
    # cv2.rectangle(shadowless_img, (x, y), (x + w, y + h), (255, 255, 255), -1)

disp_image(shadowless_img, "Rect Image")

edges = cv2.Canny(shadowless_img, 0, 257)
disp_image(edges, "Canny")

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f"Number of contours: {len(contours)}")

boxes = []  # each element is [[top-left], [bottom-right]];
hierarchy = hierarchy[0]
for currentContour, currentHierarchy in zip(contours, hierarchy):
    # currentContour = component[0]
    # currentHierarchy = component[1]
    x, y, w, h = cv2.boundingRect(currentContour)
    if currentHierarchy[3] < 0:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        boxes.append([[x, y], [x + w, y + h]])

filtered = []
for box in boxes:
    w = box[1][0] - box[0][0]
    h = box[1][1] - box[0][1]
    area = w * h
    if 100 < area < 15000 and (7 < w) and (7 < h):
        # if (5 < w < 200) and (5 < h < 200):
        filtered.append(box)
        # cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
boxes = filtered

print(len(boxes))
disp_image(img_rgb, "Bounding Boxes")

# disp_image(cv2.merge([img, shadowless_img]), "Final Image")
# disp_image(img, "Shadowless Image")

# NOTE: We need to utilize a merging capability similar to what is shown in this SO post: https://stackoverflow.com/questions/66490374/how-to-merge-nearby-bounding-boxes-opencv
# NOTE: second argument is merge_margin (affects allowed gaps)
boxes = merge(boxes, 0)
for box in boxes:
    cv2.rectangle(img_rgb, tup(box[0]), tup(box[1]), (0, 200, 0), 5)
(img_rgb_h, img_rgb_w) = img_rgb.shape[:2]
img_rgb = cv2.resize(img_rgb, (int(img_rgb_w / 3), int(img_rgb_h / 3)))
disp_image(img_rgb, "Bounding Boxes")
cv2.destroyAllWindows()
