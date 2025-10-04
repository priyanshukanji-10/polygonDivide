import cv2
import numpy as np
import random


# Function to calculate rectangle area
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


# Draw filled triangle
def draw_rectangle(img, pts):
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color())
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=1)


# Create blank image
width, height = 500, 500
img = np.zeros((height, width, 3), dtype=np.uint8)

# Generate random non-collinear triangle
while True:
    pts = [[random.randint(50, 450), random.randint(50, 450)] for _ in range(4)]
    if area(*pts) > 1e-6:
        break


def color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def divideMiddleRectangle(img, pts):
    if area(*pts) <= 1e-6:
        draw_rectangle(img, pts)
        return

    # Midpoints of each side
    m1 = [(pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2]
    m2 = [(pts[1][0] + pts[2][0]) // 2, (pts[1][1] + pts[2][1]) // 2]
    m3 = [(pts[2][0] + pts[3][0]) // 2, (pts[2][1] + pts[3][1]) // 2]
    m4 = [(pts[3][0] + pts[0][0]) // 2, (pts[3][1] + pts[0][1]) // 2]

    # The central triangle from midpoints
    draw_rectangle(img, [m1, m2, m3, m4])
    divideMiddleRectangle(img, [m1, m2, m3, m4])


# Draw the initial triangle, using a random color
draw_rectangle(img, pts)

# Recursively dividing
divideMiddleRectangle(img, pts)

cv2.imshow("Divided Triangle", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
