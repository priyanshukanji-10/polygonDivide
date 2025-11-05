import cv2
import numpy as np
import random


# Function to calculate area of a polygon
def area(*points):
    n = len(points)
    if n < 3:
        return 0  # Not a polygon

    s1 = s2 = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s1 += x1 * y2
        s2 += y1 * x2
    return abs(0.5 * (s1 - s2))


# Random color generator
def color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# Draw filled polygon
def draw_shape(img, pts):
    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], color())
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 0), thickness=1)


# Convexity check
def is_convex(pts):
    pts = np.array(pts)
    n = len(pts)
    if n < 4:
        return True  # Triangle is always convex
    signs = []
    for i in range(n):
        p0, p1, p2 = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        signs.append(np.sign(cross))
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


# Sort points around centroid
def sort_points(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return pts


# Draw recursively subdivided triangle (for visual effect only)
def divideMiddleTriangle(img, pts):
    if area(*pts) <= 1e-6:
        draw_shape(img, pts)
        return

    # Midpoints
    m1 = [(pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2]
    m2 = [(pts[1][0] + pts[2][0]) // 2, (pts[1][1] + pts[2][1]) // 2]
    m3 = [(pts[2][0] + pts[0][0]) // 2, (pts[2][1] + pts[0][1]) // 2]

    draw_shape(img, [m1, m2, m3])
    divideMiddleTriangle(img, [m1, m2, m3])


# Triangulate polygon and calculate outer triangle centroids
def triangulate(img, pts):
    n = len(pts)
    if n < 3:
        return []

    centroids = []
    ref = random.choice(pts)  # random reference vertex

    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        tri = [ref, p1, p2]

        # Centroid of outer triangle
        cx = (ref[0] + p1[0] + p2[0]) / 3
        cy = (ref[1] + p1[1] + p2[1]) / 3
        centroids.append((cx, cy))

        divideMiddleTriangle(img, tri)

        cv2.polylines(
            img,
            [np.array(tri, np.int32).reshape((-1, 1, 2))],
            isClosed=True,
            color=(0, 0, 0),
            thickness=1,
        )

    return centroids


def main():
    width, height = 500, 500
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate random convex pentagon
    while True:
        pts = [[random.randint(50, 450), random.randint(50, 450)] for _ in range(5)]
        pts = sort_points(pts)
        if area(*pts) > 1e-6 and is_convex(pts):
            break

    draw_shape(img, pts)

    # Triangulate and get centroids
    centroids = triangulate(img, pts)
    polygon = np.array(pts, np.int32)

    # Filter centroids
    filtered_centroids = []
    for cx, cy in centroids:
        # cv2.pointPolygonTest returns >0 if inside, 0 on edge, <0 outside
        if cv2.pointPolygonTest(polygon, (cx, cy), measureDist=False) > 0:
            filtered_centroids.append((cx, cy))

    for cx, cy in filtered_centroids:
        # cv2.drawMarker will automatically handle the float by rounding
        cv2.drawMarker(
            img,
            (round(cx), round(cy)),
            color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=4,
            thickness=2,
        )

    print("Outer triangle centroids:", filtered_centroids)
    divideMiddleTriangle(img, filtered_centroids)

    cv2.imshow("Pentagon Triangulation with Centroids", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
