import cv2
import numpy as np
import random


# Function to calculate area
def area(*points):
    n = len(points)
    if n < 3:
        return 0  # Not a polygon

    s1 = 0
    s2 = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]  # next vertex (wraps around)
        s1 += x1 * y2
        s2 += y1 * x2

    return abs(0.5 * (s1 - s2))


# Random color
def color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# Draw filled pentagon
def draw_pentagon(img, pts):
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color())
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=1)


# Convexity check
def is_convex(pts):
    pts = np.array(pts)
    n = len(pts)
    if n < 4:
        return True  # Triangles are always convex

    signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        signs.append(np.sign(cross))

    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


# Sorting points arround centroid
def sort_points(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return pts


def divideMiddlepentagon(img, pts):
    if area(*pts) <= 2:
        draw_pentagon(img, pts)
        return

    midpoints = []
    for i in range(len(pts)):
        x_mid = (pts[i][0] + pts[(i + 1) % len(pts)][0]) // 2
        y_mid = (pts[i][1] + pts[(i + 1) % len(pts)][1]) // 2
        midpoints.append([x_mid, y_mid])

    if not is_convex(midpoints):
        return

    draw_pentagon(img, midpoints)
    divideMiddlepentagon(img, midpoints)


# Create blank image
width, height = 500, 500
img = np.zeros((height, width, 3), dtype=np.uint8)

# Generate random convex pentagon
while True:
    pts = [[random.randint(50, 450), random.randint(50, 450)] for _ in range(5)]
    pts = sort_points(pts)
    if area(*pts) > 1e-6 and is_convex(pts):
        break

draw_pentagon(img, pts)
divideMiddlepentagon(img, pts)

cv2.imshow("Divided pentagon (Convex Only)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
