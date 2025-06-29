import cv2
import numpy as np

# --- Configuration ---
IMAGE_PATH = "scoreboard.png"
roi_points = []
drawing = False


def draw_rectangle(event, x, y, flags, param):
    """Mouse callback function to draw rectangles."""
    global roi_points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = image.copy()
            cv2.rectangle(img_copy, roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        drawing = False
        img_copy = image.copy()
        cv2.rectangle(img_copy, roi_points[0], roi_points[1], (0, 255, 0), 2)
        cv2.imshow(window_name, img_copy)


def get_roi(image, name):
    """Displays the image and prompts the user to select an ROI."""
    global roi_points
    roi_points = []

    window_title = (
        f"Draw box for '{name}'. Press 's' to save, 'r' to retry, 'q' to quit."
    )
    print("\n" + "=" * len(window_title))
    print(window_title)
    print("=" * len(window_title))

    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):  # Press 'r' to reset the drawing
            return get_roi(image, name)
        elif key == ord("s"):  # Press 's' to save the ROI
            if len(roi_points) == 2:
                p1, p2 = roi_points
                # Ensure p1 is top-left and p2 is bottom-right
                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                return (x1, y1, x2, y2)
            else:
                print("Error: Rectangle not fully drawn. Please try again.")
        elif key == ord("q"):  # Press 'q' to quit
            return None


# --- Main Execution ---
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Could not load image at '{IMAGE_PATH}'")
else:
    window_name = "ROI Selector"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)

    roi_names = [
        "map",
        "result",
        "team1_names",
        "team1_heroes",
        "team2_names",
        "team2_heroes",
    ]
    final_coords = {}

    for name in roi_names:
        coords = get_roi(image, name)
        if coords is None:
            print("Quitting...")
            break
        final_coords[name] = coords
        print(f"Saved '{name}': {coords}")

    cv2.destroyAllWindows()

    print("\n--- COPY THE DICTIONARY BELOW ---")
    print("\nfinal_roi_coords = {")
    for name, coords in final_coords.items():
        print(f"    '{name}': {coords},")
    print("}")
