import cv2
import os

# --- CONFIGURATION ---
SPRITE_SHEET_PATH = "hero_sprite_sheet.png"
OUTPUT_FOLDER = "hero_templates/"
OUTPUT_SIZE = (80, 80)  # The final dimensions (width, height)
# ---------------------

# Global variables for mouse callback
roi_points = []
drawing = False
image = None  # Will hold the main image
window_name = "Sprite Sheet Cutter"


def draw_rectangle(event, x, y, flags, param):
    """Mouse callback to draw the master rectangle."""
    global roi_points, drawing

    img_copy = image.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.rectangle(img_copy, roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_points.append((x, y))
        drawing = False
        cv2.rectangle(
            img_copy, roi_points[0], roi_points[1], (0, 0, 255), 2
        )  # Draw final box in red
        cv2.imshow(window_name, img_copy)


# --- Main Execution ---
if __name__ == "__main__":
    image = cv2.imread(SPRITE_SHEET_PATH)
    if image is None:
        print(f"Error: Could not load sprite sheet at '{SPRITE_SHEET_PATH}'")
        exit()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    # Step 1: Get the master box from the user
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_rectangle)

    print("=" * 60)
    print("INSTRUCTIONS:")
    print("1. Draw a tight box around the TOP-LEFT hero portrait.")
    print("2. When the RED box appears, you are done drawing.")
    print("3. Press 's' to SAVE and CONTINUE, 'r' to REDRAW, or 'q' to QUIT.")
    print("=" * 60)

    cv2.imshow(window_name, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if len(roi_points) == 2:
                break  # Exit the loop to proceed
            else:
                print("Error: Please draw a complete box first.")
        elif key == ord("r"):
            # Reset and redraw
            roi_points = []
            cv2.imshow(window_name, image)
            print("Selection cleared. Please redraw the box.")
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()

    # This part now only runs after a successful selection
    cv2.destroyAllWindows()

    p1, p2 = roi_points
    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
    master_w, master_h = x2 - x1, y2 - y1

    # --- THIS IS THE FIX ---
    # Check if the drawn box is valid before continuing
    if master_w == 0 or master_h == 0:
        print("\nError: The selection has zero width or height. Please try again.")
        exit()

    try:
        cols = int(input("\nEnter the number of COLUMNS in the grid: "))
        rows = int(input("Enter the number of ROWS in the grid: "))
    except ValueError:
        print("Invalid input. Please enter numbers.")
        exit()

    print(f"\nGrid defined: {cols} columns by {rows} rows.")
    print(f"Master icon size: {master_w}x{master_h} pixels.")
    print("Starting extraction...")

    count = 0
    for r in range(rows):
        for c in range(cols):
            count += 1
            current_x = x1 + c * master_w
            current_y = y1 + r * master_h
            cropped_icon = image[
                current_y : current_y + master_h, current_x : current_x + master_w
            ]

            # Check if the cropped icon is valid before resizing
            if cropped_icon.size == 0:
                print(
                    f"Warning: Skipping icon at row {r+1}, col {c+1} due to invalid crop."
                )
                continue

            resized_icon = cv2.resize(
                cropped_icon, OUTPUT_SIZE, interpolation=cv2.INTER_AREA
            )

            window_title = f"Icon {count}/{rows*cols}"
            cv2.imshow(window_title, resized_icon)

            hero_name = input(
                f"What is the name for Icon {count}? (e.g., 'dva', or 'skip'): "
            )

            cv2.destroyWindow(window_title)

            if hero_name.lower() == "skip":
                print("  -> Skipping.")
                continue

            if hero_name:
                output_path = os.path.join(OUTPUT_FOLDER, f"{hero_name}.png")
                cv2.imwrite(output_path, resized_icon)
                print(f"  -> Saved {OUTPUT_SIZE} icon to {output_path}")

    print("\nProcessing complete!")
