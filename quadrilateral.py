import cv2
import numpy as np
import random


# Function to calculate area
def area(p1, p2, p3, p4):
    return abs(
        0.5
        * (
            p1[0] * (p2[1] - p4[1])
            + p2[0] * (p3[1] - p1[1])
            + p3[0] * (p4[1] - p2[1])
            + p4[0] * (p1[1] - p3[1])
        )
    )


# Random color
def color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# Draw filled quadrilateral
def draw_quadrilateral(img, pts):
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color())
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=1)


# Convexity check
def is_convex(pts):
    pts = np.array(pts)
    pts = pts.reshape((-1, 2))
    # Compute cross products to check all turns go in same direction
    signs = []
    for i in range(4):
        p0, p1, p2 = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        signs.append(np.sign(cross))
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def divideMiddlequadrilateral(img, pts):
    if area(*pts) <= 2:
        draw_quadrilateral(img, pts)
        return

    midpoints = []
    for i in range(len(pts)):
        x_mid = (pts[i][0] + pts[(i + 1) % len(pts)][0]) // 2
        y_mid = (pts[i][1] + pts[(i + 1) % len(pts)][1]) // 2
        midpoints.append([x_mid, y_mid])

    if not is_convex(midpoints):
        return

    draw_quadrilateral(img, midpoints)
    divideMiddlequadrilateral(img, midpoints)


# Create blank image
width, height = 500, 500
img = np.zeros((height, width, 3), dtype=np.uint8)

# Generate random convex quadrilateral
while True:
    pts = [[random.randint(50, 450), random.randint(50, 450)] for _ in range(4)]
    if area(*pts) > 1e-6 and is_convex(pts):
        break

draw_quadrilateral(img, pts)
divideMiddlequadrilateral(img, pts)

cv2.imshow("Divided quadrilateral (Convex Only)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
