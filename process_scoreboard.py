import cv2
import numpy as np
import os
import pytesseract
import json

KNOWN_PLAYERS = ["RETRAC", "JISOO", "SAMPHIL"]
# ----------------------------------------------------

# --- CONFIGURATION ---
SCOREBOARD_PATH = "scoreboards/samoa_score.png"
HERO_TEMPLATES_PATH = "hero_templates/"
MAP_TEMPLATES_PATH = "map_templates/"
NAME_TEMPLATES_PATH = "name_templates/"  # --- NEW ---

# --- PARAMETERS ---
MAP_CONFIDENCE_THRESHOLD = 0.80
HERO_DETECTION_THRESHOLD = 0.70
NAME_DETECTION_THRESHOLD = 0.85


def find_heroes_in_roi(roi, hero_templates, threshold):
    """
    Finds heroes and prints the confidence score for every check.
    """
    found_heroes = []
    roi_h, roi_w = roi.shape[:2]

    # Instead of finding all locations, we'll find the *best* single match for each template.
    for name, template in hero_templates.items():
        if template is None:
            continue

        templ_h, templ_w = template.shape[:2]
        if templ_h > roi_h or templ_w > roi_w:
            continue

        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)

        print(f"  - Checking for {name:<12} | Best match score: {max_val:.2f}")

        if max_val >= threshold:
            is_new = all(
                abs(max_loc[0] - ex) > 20 or abs(max_loc[1] - ey) > 20
                for _, ex, ey, _ in found_heroes
            )
            if is_new:
                found_heroes.append((name, max_loc[0], max_loc[1], max_val))
                print(
                    f"    └──> DETECTED {name} at ({max_loc[0]}, {max_loc[1]}) with score {max_val:.2f}"
                )
    return found_heroes


def find_known_players_in_roi(roi, name_templates, threshold):
    """
    Finds all occurrences of known player names using template matching.
    """
    found_players = []
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    for name, template in name_templates.items():
        if template is None:
            print(f"  - Skipping template for '{name}' as it could not be loaded.")
            continue

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template_gray.shape[::-1]

        res = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Find the single best match score for this template to print it, just like for heroes/maps.
        _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)
        print(f"  - Checking for '{name:<12}' | Best match score: {max_val:.2f}")

        locs = np.where(res >= threshold)

        detections = []
        for pt in zip(*locs[::-1]):  # switch x and y
            score = res[pt[1], pt[0]]
            detections.append((pt[0], pt[1], score))

        # Non-maximum suppression: If multiple detections are too close, keep only the best one.
        suppressed_detections = []
        detections.sort(key=lambda x: x[2], reverse=True)

        for x, y, score in detections:
            is_close = False
            for sx, sy, _ in suppressed_detections:
                if abs(x - sx) < w * 0.5 and abs(y - sy) < h * 0.5:
                    is_close = True
                    break
            if not is_close:
                suppressed_detections.append((x, y, score))

        if suppressed_detections:
            for x, y, score in suppressed_detections:
                # Store the name, its y-coordinate (for sorting), and the x-coordinate
                found_players.append({"name": name, "y": y, "x": x, "score": score})
                print(
                    f"    └──> DETECTED {name} at ({x}, {y}) with score {score:.2f} (Threshold: {threshold})"
                )

    return found_players


def find_best_map_match(map_roi, map_templates, threshold):
    """
    Finds the single best map match and prints its confidence.
    """
    best_match_score = -1
    best_match_name = "Unknown"

    print("\n--- MAP DETECTION ---")
    for name, template in map_templates.items():
        if template is None:
            continue
        res = cv2.matchTemplate(map_roi, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        print(f"  - Checking for {name:<20} | Confidence: {max_val:.2f}")
        if max_val > best_match_score:
            best_match_score = max_val
            best_match_name = name

    if best_match_score >= threshold:
        print(
            f"└──> Best Match Found: {best_match_name} (Score: {best_match_score:.2f})"
        )
        return best_match_name
    else:
        print(
            f"└──> No map found above threshold {threshold}. Best attempt was {best_match_name} (Score: {best_match_score:.2f})"
        )
        return "Unknown"


def analyze_scoreboard(scoreboard_img_path):
    if not os.path.exists(scoreboard_img_path):
        print(f"Error: Scoreboard image not found at {scoreboard_img_path}")
        return
    scoreboard_img = cv2.imread(scoreboard_img_path)
    if scoreboard_img is None:
        print(f"Error: Could not read image file {scoreboard_img_path}")
        return

    # Load templates
    map_templates = {
        os.path.basename(p).split(".")[0]: cv2.imread(
            os.path.join(MAP_TEMPLATES_PATH, p)
        )
        for p in os.listdir(MAP_TEMPLATES_PATH)
        if p.endswith(".png")
    }
    hero_templates = {
        os.path.basename(p).split(".")[0]: cv2.imread(
            os.path.join(HERO_TEMPLATES_PATH, p)
        )
        for p in os.listdir(HERO_TEMPLATES_PATH)
        if p.endswith(".png")
    }

    name_templates = {
        name: cv2.imread(os.path.join(NAME_TEMPLATES_PATH, f"{name}.png"))
        for name in KNOWN_PLAYERS
    }

    # Define Regions of Interest (ROIs)
    # Note: These might need slight tweaking
    roi_coords = {
        "map": (1515, 291, 2205, 738),
        "result": (1556, 773, 1770, 847),
        "team1_names": (479, 336, 713, 743),
        "team1_heroes": (390, 328, 477, 744),
        "team2_names": (477, 868, 778, 1279),
        "team2_heroes": (385, 849, 483, 1288),
    }

    # Crop ROIs from the main image
    ROI_MAP = scoreboard_img[
        roi_coords["map"][1] : roi_coords["map"][3],
        roi_coords["map"][0] : roi_coords["map"][2],
    ]
    ROI_HEROES_1 = scoreboard_img[
        roi_coords["team1_heroes"][1] : roi_coords["team1_heroes"][3],
        roi_coords["team1_heroes"][0] : roi_coords["team1_heroes"][2],
    ]
    ROI_HEROES_2 = scoreboard_img[
        roi_coords["team2_heroes"][1] : roi_coords["team2_heroes"][3],
        roi_coords["team2_heroes"][0] : roi_coords["team2_heroes"][2],
    ]
    ROI_RESULT = scoreboard_img[
        roi_coords["result"][1] : roi_coords["result"][3],
        roi_coords["result"][0] : roi_coords["result"][2],
    ]
    ROI_TEAM1_NAMES = scoreboard_img[
        roi_coords["team1_names"][1] : roi_coords["team1_names"][3],
        roi_coords["team1_names"][0] : roi_coords["team1_names"][2],
    ]
    ROI_TEAM2_NAMES = scoreboard_img[
        roi_coords["team2_names"][1] : roi_coords["team2_names"][3],
        roi_coords["team2_names"][0] : roi_coords["team2_names"][2],
    ]

    # --- 1. DETECT MAP ---
    detected_map = find_best_map_match(ROI_MAP, map_templates, MAP_CONFIDENCE_THRESHOLD)

    # --- 2. DETECT HEROES ---
    print("\n--- TEAM 1 HERO DETECTION ---")
    team1_heroes_found = find_heroes_in_roi(
        ROI_HEROES_1, hero_templates, HERO_DETECTION_THRESHOLD
    )
    print("\n--- TEAM 2 HERO DETECTION ---")
    team2_heroes_found = find_heroes_in_roi(
        ROI_HEROES_2, hero_templates, HERO_DETECTION_THRESHOLD
    )

    # Sort by y-coordinate to match player list order
    team1_heroes_sorted = sorted(team1_heroes_found, key=lambda item: item[2])
    team2_heroes_sorted = sorted(team2_heroes_found, key=lambda item: item[2])

    # --- 3. DETECT GAME RESULT ---
    print("\n--- TEXT RECOGNITION (OCR for Game Result) ---")
    try:
        result_text = (
            pytesseract.image_to_string(ROI_RESULT, config="--psm 8").strip().upper()
        )
        match_result = (
            "VICTORY"
            if "VICTORY" in result_text
            else "DEFEAT" if "DEFEAT" in result_text else "UNKNOWN"
        )
        print(f"  - Game Result Detected: {match_result} (Raw OCR: '{result_text}')")
    except Exception as e:
        print(f"\n--- OCR FAILED --- \nAn error occurred: {e}")
        match_result = "OCR_FAILED"

    # --- 4. DETECT PLAYER NAMES ---
    print("\n--- PLAYER NAME DETECTION (Template Matching) ---")
    print("--- Detecting in Team 1 ---")
    team1_players_found = find_known_players_in_roi(
        ROI_TEAM1_NAMES, name_templates, NAME_DETECTION_THRESHOLD
    )
    print("\n--- Detecting in Team 2 ---")
    team2_players_found = find_known_players_in_roi(
        ROI_TEAM2_NAMES, name_templates, NAME_DETECTION_THRESHOLD
    )

    # Sort players by their vertical position (y-coordinate)
    team1_players_sorted = sorted(team1_players_found, key=lambda p: p["y"])
    team2_players_sorted = sorted(team2_players_found, key=lambda p: p["y"])

    # Extract just the names in the correct order
    team1_players = [p["name"] for p in team1_players_sorted]
    team2_players = [p["name"] for p in team2_players_sorted]

    # --- 5. ASSEMBLE FINAL DATA ---
    final_data = {"map": detected_map, "result": match_result, "team1": [], "team2": []}

    print("\n--- DATA VALIDATION ---")
    print(
        f"Found {len(team1_players)} known player names and {len(team1_heroes_sorted)} hero icons for Team 1."
    )
    print(
        f"Found {len(team2_players)} known player names and {len(team2_heroes_sorted)} hero icons for Team 2."
    )

    print("\n--- Pairing Players with Heroes ---")

    for player in team1_players_sorted:
        # Find the hero icon that is vertically closest to this player's name
        closest_hero = min(team1_heroes_sorted, key=lambda h: abs(h[2] - player["y"]))
        final_data["team1"].append(
            {"player_name": player["name"], "hero": closest_hero[0].title()}
        )
        print(
            f"  - Team 1: Paired {player['name']} (y={player['y']}) with {closest_hero[0]} (y={closest_hero[2]})"
        )

    for player in team2_players_sorted:
        closest_hero = min(team2_heroes_sorted, key=lambda h: abs(h[2] - player["y"]))
        final_data["team2"].append(
            {"player_name": player["name"], "hero": closest_hero[0].title()}
        )
        print(
            f"  - Team 2: Paired {player['name']} (y={player['y']}) with {closest_hero[0]} (y={closest_hero[2]})"
        )

    print("\n\n--- EXTRACTION COMPLETE ---")
    print("Final structured data:")
    print(json.dumps(final_data, indent=2))


if __name__ == "__main__":
    analyze_scoreboard(SCOREBOARD_PATH)
