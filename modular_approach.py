import math
import cv2
import numpy as np


def doOpt(b, f):
    if b:
        f()


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


def merge_only_overlaps(boxes, img=None):
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
            disp_image(copy, "Copy")

        # loop through boxes
        index = len(boxes) - 1
        while index >= 0:
            # grab current box
            curr = boxes[index]
            # add margin
            tl = curr[0][:]
            br = curr[1][:]
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
            disp_image(copy, "Copy")

        # loop through boxes
        index = len(boxes) - 1
        while index >= 0:
            # grab current box
            curr = boxes[index]
            # add margin
            tl = curr[0][:]
            br = curr[1][:]
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

        # loop through boxes
        index = len(boxes) - 1
        while index >= 0:
            # grab current box
            curr = boxes[index]
            # We want to have a directional merge, so we are more likely to merge boxes that are in the same dimension as the current longest dimension of the box
            # print(curr)
            # add margin
            tl = curr[0][:]
            x1, x2 = curr[0][0], curr[1][0]
            y1, y2 = curr[0][1], curr[1][1]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            # h_modifier = 0.2 * h if h > (w * 1.3) else 1
            # w_modifier = 0.2 * w if w > (h * 1.3) else 1
            h_modifier = 2 * (h / w) ** 1.5 if h > (w * 1.3) else 1
            w_modifier = 2 * (w / h) ** 1.5 if w > (h * 1.3) else 1
            # punish small bits as they should not merge as easily
            h_modifier = 1 if area < 250 else h_modifier
            w_modifier = 1 if area < 250 else w_modifier
            # h_modifier *= math.log2(area) + 1
            # w_modifier *= math.log2(area) + 1
            br = curr[1][:]
            tl[0] -= merge_margin * w_modifier
            tl[1] -= merge_margin * h_modifier
            br[0] += merge_margin * w_modifier
            br[1] += merge_margin * h_modifier
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


def remove_shadows(img: np.ndarray) -> np.ndarray:
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

    # Apply binary thresholding to segment shadow regions
    return img1


def remove_center(orig_img: np.ndarray, shadowless_img: np.ndarray) -> None:
    my_img = orig_img.copy()

    my_img = cv2.cvtColor(my_img, cv2.COLOR_BGR2GRAY)
    # apply threshold to mask IC housing
    blur2 = cv2.bilateralFilter(my_img, 9, 75, 75)
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


def get_edges(shadowless_img: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(shadowless_img, 0, 257)
    return edges


def get_contours(edge_img: np.ndarray) -> list:
    contours, hierarchy = cv2.findContours(
        edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    print(f"Number of contours: {len(contours)}")

    boxes = []  # each element is [[top-left], [bottom-right]];
    hierarchy = hierarchy[0]
    for currentContour, currentHierarchy in zip(contours, hierarchy):
        x, y, w, h = cv2.boundingRect(currentContour)
        if currentHierarchy[3] < 0:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            boxes.append([[x, y], [x + w, y + h]])

    filtered = []
    for box in boxes:
        w = box[1][0] - box[0][0]
        h = box[1][1] - box[0][1]
        area = w * h
        if 120 < area < 15000 and (8 < w) and (8 < h):
            # if (5 < w < 200) and (5 < h < 200):
            filtered.append(box)
            # cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    boxes = sorted(
        filtered,
        key=(lambda x: (x[1][0] - x[0][0]) * (x[1][1] - x[0][1])),
        reverse=True,
    )
    return boxes


def draw_boxes_color(orig_img: np.ndarray, boxes: list) -> np.ndarray:
    my_img = orig_img.copy()
    for box in boxes:
        cv2.rectangle(my_img, tup(box[0]), tup(box[1]), box[2], 5)
    return my_img


def draw_boxes(orig_img: np.ndarray, boxes: list) -> np.ndarray:
    my_img = orig_img.copy()
    for box in boxes:
        cv2.rectangle(my_img, tup(box[0]), tup(box[1]), (0, 200, 0), 5)
    return my_img


def filter_boxes(boxes: list) -> list:
    AREA_FILTER = 1000
    filtered_boxes = []
    for box in boxes:
        [x1, y1], [x2, y2] = box
        if (x2 - x1) * (y2 - y1) >= AREA_FILTER:
            filtered_boxes.append(box)
    return filtered_boxes


def split_boxes(img_center, boxes: list) -> tuple[list, list, list, list, list]:
    center_x, center_y = img_center
    # First split based on which direction they are point
    up_downs = []
    left_rights = []
    ambiguous = []
    for box in boxes:
        [x1, y1], [x2, y2] = box
        w = x2 - x1
        h = y2 - y1
        if w > h * 1.3:
            left_rights.append([box[0], box[1], (0, 0, 255)])
        elif h > w * 1.3:
            up_downs.append([box[0], box[1], (255, 0, 0)])
        else:
            ambiguous.append([box[0], box[1], (0, 255, 0)])
    # Split the up_downs into up and down
    up_downs = sorted(up_downs, key=lambda x: x[0][1])
    ups = filter(lambda x: x[1][1] < center_y, up_downs)
    # change up colors to yellow
    ups = list(map(lambda x: [x[0], x[1], (0, 255, 255)], ups))
    downs = filter(lambda x: x[0][1] > center_y, up_downs)
    # change down colors to red
    downs = list(map(lambda x: [x[0], x[1], (255, 0, 0)], downs))
    # split the left_rights into left and right
    left_rights = sorted(left_rights, key=lambda x: x[0][0])
    lefts = filter(lambda x: x[1][0] < center_x, left_rights)
    # change left colors to light blue
    lefts = list(map(lambda x: [x[0], x[1], (255, 255, 0)], lefts))
    rights = filter(lambda x: x[0][0] > center_x, left_rights)
    # change right colors to pink
    rights = list(map(lambda x: [x[0], x[1], (255, 0, 255)], rights))
    return ups, downs, lefts, rights, ambiguous


def run_img(img_path: str, disp_all: bool) -> np.ndarray:
    # NOTE: Throughout the functions that accept images will expect them to be in BGR form, and return in BGR form.
    # Everyone will responsible for their own conversions
    orig_img = cv2.imread(input_image)

    doOpt(disp_all, lambda _: disp_image(orig_img))

    # Remove shadows from the image
    shadowless_img = remove_shadows(orig_img)
    doOpt(disp_all, lambda _: disp_image(shadowless_img))

    # Remove the center from the image
    remove_center(orig_img, shadowless_img)
    doOpt(disp_all, lambda _: disp_image(shadowless_img))

    # Detect edges
    edges = get_edges(shadowless_img)
    doOpt(disp_all, lambda _: disp_image(edges))

    # get initial boxes
    boxes = get_contours(edges)

    # Merge boxes
    boxes = merge(boxes, 1)

    # Draw on boxes
    boxed_img = draw_boxes(orig_img, boxes)
    doOpt(disp_all, lambda _: disp_image(boxed_img))

    # Filter boxes
    filt_boxes = filter_boxes(boxes)

    # Split into a N, E, S, W groups
    # find center_x and center_y
    img_h, img_w = orig_img.shape[:2]
    center_x = img_w // 2
    center_y = img_h // 2
    ups, downs, lefts, rights, ambiguous = split_boxes([center_x, center_y], filt_boxes)

    # Draw new boxes
    filt_boxed_img = draw_boxes_color(
        orig_img, ups + downs + lefts + rights + ambiguous
    )
    doOpt(disp_all, lambda _: disp_image(filt_boxed_img))

    return filt_boxed_img


# Example usage
image_corpus = [
    "./pins_images/A-J-28SOP-01B-SM.png",
    "./pins_images/A-D-64QFP-14B-SM.png",
    "./pins_images/A-D-64QFP-15B-SM.png",
    "./pins_images/C-T-28SOP-04F-SM.png",
]

for input_image in image_corpus:
    disp_image(run_img(input_image, False))

cv2.destroyAllWindows()