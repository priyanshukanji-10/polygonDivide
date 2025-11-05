import cv2
import numpy as np
import random
import pandas as pd
import os
from scipy.spatial import Delaunay


# Function to calculate area of a polygon
def area(*points):
    n = len(points)
    if n < 3:
        return 0

    s1 = s2 = 0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s1 += x1 * y2
        s2 += y1 * x2
    return abs(0.5 * (s1 - s2))


# Classical centroid calculation
def classical_centroid(pts):
    n = len(pts)
    if n < 3:
        return None

    A = 0
    cx_sum = 0
    cy_sum = 0

    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        A += cross
        cx_sum += (x1 + x2) * cross
        cy_sum += (y1 + y2) * cross

    A = A / 2
    if abs(A) < 1e-10:
        return None

    cx = cx_sum / (6 * A)
    cy = cy_sum / (6 * A)

    return (cx, cy)


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
        return True
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


# Draw recursively subdivided triangle
def divideMiddleTriangle(img, pts):
    if area(*pts) <= 1e-6:
        draw_shape(img, pts)
        return

    m1 = [(pts[0][0] + pts[1][0]) // 2, (pts[0][1] + pts[1][1]) // 2]
    m2 = [(pts[1][0] + pts[2][0]) // 2, (pts[1][1] + pts[2][1]) // 2]
    m3 = [(pts[2][0] + pts[0][0]) // 2, (pts[2][1] + pts[0][1]) // 2]

    draw_shape(img, [m1, m2, m3])
    divideMiddleTriangle(img, [m1, m2, m3])


# Method 1: Random vertex triangulation (original)
def triangulate_random(img, pts):
    n = len(pts)
    if n < 3:
        return []

    centroids = []
    ref = random.choice(pts)

    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        tri = [ref, p1, p2]

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


# Method 2: Center-point triangulation (uses classical centroid)
def triangulate_center(img, pts, center):
    n = len(pts)
    if n < 3:
        return []

    centroids = []
    ref = [center[0], center[1]]

    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        tri = [ref, p1, p2]

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


# Method 3: Fan triangulation (from first vertex)
def triangulate_fan(img, pts):
    n = len(pts)
    if n < 3:
        return []

    centroids = []
    ref = pts[0]  # Always use first vertex

    for i in range(1, n - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        tri = [ref, p1, p2]

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


# Method 4: Delaunay triangulation
def triangulate_delaunay(img, pts):
    n = len(pts)
    if n < 3:
        return []

    points = np.array(pts)
    tri = Delaunay(points)

    centroids = []

    for simplex in tri.simplices:
        p1, p2, p3 = points[simplex]

        # Calculate centroid
        cx = (p1[0] + p2[0] + p3[0]) / 3
        cy = (p1[1] + p2[1] + p3[1]) / 3
        centroids.append((cx, cy))

        # Draw triangle
        triangle = [p1.tolist(), p2.tolist(), p3.tolist()]
        divideMiddleTriangle(img, triangle)
        cv2.polylines(
            img,
            [np.array(triangle, np.int32).reshape((-1, 1, 2))],
            isClosed=True,
            color=(0, 0, 0),
            thickness=1,
        )

    return centroids


# Method 5: Ear clipping triangulation
def triangulate_ear_clipping(img, pts):
    n = len(pts)
    if n < 3:
        return []

    centroids = []
    remaining = pts.copy()

    def is_ear(prev_pt, curr_pt, next_pt, polygon):
        # Check if triangle formed by these 3 points is an ear
        tri = [prev_pt, curr_pt, next_pt]

        # Check if it's a convex vertex
        v1 = [curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]]
        v2 = [next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1]]
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        if cross <= 0:  # Reflex angle
            return False

        # Check if any other point is inside the triangle
        for pt in polygon:
            if pt == prev_pt or pt == curr_pt or pt == next_pt:
                continue
            if point_in_triangle(pt, tri):
                return False

        return True

    def point_in_triangle(p, tri):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, tri[0], tri[1])
        d2 = sign(p, tri[1], tri[2])
        d3 = sign(p, tri[2], tri[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    while len(remaining) > 3:
        for i in range(len(remaining)):
            prev_pt = remaining[(i - 1) % len(remaining)]
            curr_pt = remaining[i]
            next_pt = remaining[(i + 1) % len(remaining)]

            if is_ear(prev_pt, curr_pt, next_pt, remaining):
                # Found an ear, clip it
                tri = [prev_pt, curr_pt, next_pt]
                cx = (prev_pt[0] + curr_pt[0] + next_pt[0]) / 3
                cy = (prev_pt[1] + curr_pt[1] + next_pt[1]) / 3
                centroids.append((cx, cy))

                divideMiddleTriangle(img, tri)
                cv2.polylines(
                    img,
                    [np.array(tri, np.int32).reshape((-1, 1, 2))],
                    isClosed=True,
                    color=(0, 0, 0),
                    thickness=1,
                )

                # Remove the ear tip
                remaining.pop(i)
                break

    # Add the last triangle
    if len(remaining) == 3:
        cx = sum(p[0] for p in remaining) / 3
        cy = sum(p[1] for p in remaining) / 3
        centroids.append((cx, cy))

        divideMiddleTriangle(img, remaining)
        cv2.polylines(
            img,
            [np.array(remaining, np.int32).reshape((-1, 1, 2))],
            isClosed=True,
            color=(0, 0, 0),
            thickness=1,
        )

    return centroids


def filter_centroids(centroids, polygon):
    """Filter centroids that are inside the polygon"""
    filtered = []
    for cx, cy in centroids:
        if cv2.pointPolygonTest(polygon, (cx, cy), measureDist=False) > 0:
            filtered.append((cx, cy))
    return filtered


def calculate_centroid_from_list(centroids):
    """Calculate average centroid from list"""
    if not centroids:
        return None
    avg_x = sum(c[0] for c in centroids) / len(centroids)
    avg_y = sum(c[1] for c in centroids) / len(centroids)
    return (avg_x, avg_y)


def append_to_excel(filename, df_comparison, df_details, run_number):
    """Append data to existing Excel file or create new one"""

    df_comparison["Run"] = run_number
    df_details["Run"] = run_number

    df_comparison = df_comparison[
        ["Run"] + [col for col in df_comparison.columns if col != "Run"]
    ]
    df_details = df_details[
        ["Run"] + [col for col in df_details.columns if col != "Run"]
    ]

    if os.path.exists(filename):
        with pd.ExcelWriter(
            filename, engine="openpyxl", mode="a", if_sheet_exists="overlay"
        ) as writer:
            existing_comparison = pd.read_excel(
                filename, sheet_name="Method_Comparison"
            )
            existing_details = pd.read_excel(filename, sheet_name="Detailed_Results")

            updated_comparison = pd.concat(
                [existing_comparison, df_comparison], ignore_index=True
            )
            updated_details = pd.concat(
                [existing_details, df_details], ignore_index=True
            )

            writer.book.remove(writer.book["Method_Comparison"])
            writer.book.remove(writer.book["Detailed_Results"])

            updated_comparison.to_excel(
                writer, sheet_name="Method_Comparison", index=False
            )
            updated_details.to_excel(writer, sheet_name="Detailed_Results", index=False)
    else:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df_comparison.to_excel(writer, sheet_name="Method_Comparison", index=False)
            df_details.to_excel(writer, sheet_name="Detailed_Results", index=False)


def main():
    width, height = 500, 500

    # Choose which method to visualize (change this to test different methods)
    method_choice = input(
        "Choose triangulation method:\n1. Random Vertex\n2. Center Point (Best)\n3. Fan\n4. Delaunay\n5. Ear Clipping\nEnter choice (1-5): "
    )

    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate random convex pentagon
    while True:
        pts = [[random.randint(50, 450), random.randint(50, 450)] for _ in range(5)]
        pts = sort_points(pts)
        if area(*pts) > 1e-6 and is_convex(pts):
            break

    draw_shape(img, pts)

    # Calculate classical centroid (ground truth)
    classical_cent = classical_centroid(pts)
    print(f"\n{'='*60}")
    print(
        f"Classical Centroid (Ground Truth): ({classical_cent[0]:.2f}, {classical_cent[1]:.2f})"
    )
    print(f"{'='*60}\n")

    polygon = np.array(pts, np.int32)

    # Apply chosen method for visualization
    method_map = {
        "1": ("Random Vertex", lambda: triangulate_random(img, pts)),
        "2": ("Center Point", lambda: triangulate_center(img, pts, classical_cent)),
        "3": ("Fan (First Vertex)", lambda: triangulate_fan(img, pts)),
        "4": ("Delaunay", lambda: triangulate_delaunay(img, pts)),
        "5": ("Ear Clipping", lambda: triangulate_ear_clipping(img, pts)),
    }

    if method_choice in method_map:
        chosen_name, chosen_func = method_map[method_choice]
        print(f"Visualizing: {chosen_name}\n")
        centroids = chosen_func()
    else:
        print("Invalid choice, using Center Point method")
        chosen_name = "Center Point"
        centroids = triangulate_center(img, pts, classical_cent)

    # Calculate results for ALL methods (for Excel comparison)
    all_methods = {
        "Random Vertex": triangulate_random(
            np.zeros((height, width, 3), dtype=np.uint8), pts
        ),
        "Center Point": triangulate_center(
            np.zeros((height, width, 3), dtype=np.uint8), pts, classical_cent
        ),
        "Fan (First Vertex)": triangulate_fan(
            np.zeros((height, width, 3), dtype=np.uint8), pts
        ),
        "Delaunay": triangulate_delaunay(
            np.zeros((height, width, 3), dtype=np.uint8), pts
        ),
        "Ear Clipping": triangulate_ear_clipping(
            np.zeros((height, width, 3), dtype=np.uint8), pts
        ),
    }

    results = []
    detail_results = []

    for method_name, method_centroids in all_methods.items():
        filtered = filter_centroids(method_centroids, polygon)
        avg_cent = calculate_centroid_from_list(filtered)

        if avg_cent:
            distance = np.sqrt(
                (classical_cent[0] - avg_cent[0]) ** 2
                + (classical_cent[1] - avg_cent[1]) ** 2
            )
            polygon_size = np.sqrt(area(*pts))
            relative_error = (distance / polygon_size) * 100 if polygon_size > 0 else 0

            results.append(
                {
                    "Method": method_name,
                    "Centroid_X": avg_cent[0],
                    "Centroid_Y": avg_cent[1],
                    "Distance_from_Classical": distance,
                    "Relative_Error_Percent": relative_error,
                    "Num_Triangles": len(method_centroids),
                    "Num_Inside": len(filtered),
                }
            )

            print(f"{method_name}:")
            print(f"  Centroid: ({avg_cent[0]:.2f}, {avg_cent[1]:.2f})")
            print(f"  Distance: {distance:.4f} pixels")
            print(f"  Relative Error: {relative_error:.4f}%")
            print(f"  Triangles: {len(method_centroids)} (Inside: {len(filtered)})\n")

            # Store individual centroids for detailed sheet
            for i, (cx, cy) in enumerate(filtered):
                detail_results.append(
                    {
                        "Method": method_name,
                        "Triangle_ID": i + 1,
                        "Centroid_X": cx,
                        "Centroid_Y": cy,
                    }
                )

    # Add classical centroid to results
    results.insert(
        0,
        {
            "Method": "Classical (Ground Truth)",
            "Centroid_X": classical_cent[0],
            "Centroid_Y": classical_cent[1],
            "Distance_from_Classical": 0,
            "Relative_Error_Percent": 0,
            "Num_Triangles": 0,
            "Num_Inside": 0,
        },
    )

    # Draw markers for visualized method
    filtered = filter_centroids(centroids, polygon)
    avg_cent = calculate_centroid_from_list(filtered)

    # Draw classical centroid (green star)
    cv2.drawMarker(
        img,
        (round(classical_cent[0]), round(classical_cent[1])),
        color=(0, 255, 0),
        markerType=cv2.MARKER_STAR,
        markerSize=10,
        thickness=2,
    )
    cv2.putText(
        img,
        "Classical",
        (round(classical_cent[0]) + 12, round(classical_cent[1]) - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    # Draw triangle centroids (red crosses)
    for cx, cy in filtered:
        cv2.drawMarker(
            img,
            (round(cx), round(cy)),
            color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=5,
            thickness=2,
        )

    # Draw average centroid (magenta diamond)
    if avg_cent:
        cv2.drawMarker(
            img,
            (round(avg_cent[0]), round(avg_cent[1])),
            color=(255, 0, 255),
            markerType=cv2.MARKER_DIAMOND,
            markerSize=10,
            thickness=2,
        )
        cv2.putText(
            img,
            chosen_name[:8],
            (round(avg_cent[0]) + 12, round(avg_cent[1]) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )

    # Determine run number
    filename = "triangulation_comparison.xlsx"
    if os.path.exists(filename):
        existing_data = pd.read_excel(filename, sheet_name="Method_Comparison")
        run_number = (
            existing_data["Run"].max() + 1 if "Run" in existing_data.columns else 1
        )
    else:
        run_number = 1

    # Create DataFrames and save
    df_comparison = pd.DataFrame(results)
    df_details = pd.DataFrame(detail_results)

    append_to_excel(filename, df_comparison, df_details, run_number)

    print(f"{'='*60}")
    print(f"✓ Data appended to '{filename}' as Run #{run_number}")
    print(f"{'='*60}\n")

    # Find best method
    best_method = min(
        [r for r in results if r["Method"] != "Classical (Ground Truth)"],
        key=lambda x: x["Distance_from_Classical"],
    )
    print(
        f"🏆 Best Method: {best_method['Method']} (Error: {best_method['Relative_Error_Percent']:.4f}%)"
    )

    cv2.imshow(f"Pentagon Triangulation - {chosen_name}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
