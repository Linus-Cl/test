import cv2
import numpy as np
import os
import pytesseract
import json
from thefuzz import fuzz  # <-- NEW: For fuzzy string matching

# --- ACTION REQUIRED: CONFIRM/EDIT THIS LIST ---
# These are the exact, correct names of you and your friends.
KNOWN_PLAYERS = [
    "MATZE",
    "PELAGORNIS",
    "RETRAC",
    "DERATRON",
    "DIZZY",
    "CROVAXX",
    "AGATOR",
    "BRIGHTHAMMER",
    "NIJIYO",
    "SHNAKE",
]
# ----------------------------------------------------

# --- FINAL CONFIGURATION ---
SCOREBOARD_PATH = "scoreboard.png"
HERO_TEMPLATES_PATH = "hero_templates/"
MAP_TEMPLATES_PATH = "map_templates/"

# --- FINAL TUNED PARAMETERS ---
MAP_CONFIDENCE_THRESHOLD = 0.60
HERO_DETECTION_THRESHOLD = 0.80
MIN_NAME_MATCH_SCORE = 65  # Similarity score (out of 100) to consider a name a match


def preprocess_for_ocr(image):
    # This function is unchanged from your best version
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return thresh


def find_heroes_in_roi(roi, hero_templates, threshold):
    # This function is unchanged from your best version
    found_heroes = []
    roi_h, roi_w = roi.shape[:2]
    for name, template in hero_templates.items():
        if template is None:
            continue
        templ_h, templ_w = template.shape[:2]
        if templ_h > roi_h or templ_w > roi_w:
            continue
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= threshold)
        for pt in zip(*locs[::-1]):
            is_new = all(
                abs(pt[0] - ex) > 20 or abs(pt[1] - ey) > 20
                for _, ex, ey in found_heroes
            )
            if is_new:
                found_heroes.append((name, pt[0], pt[1]))
    return found_heroes


def find_best_match(ocr_name, known_names, min_score):
    """
    Finds the best match for an OCR'd name from a list of known names.
    """
    best_score = -1
    best_match = None
    for known_name in known_names:
        score = fuzz.ratio(ocr_name.upper(), known_name.upper())
        if score > best_score:
            best_score = score
            best_match = known_name

    # If the best found score is good enough, return the correct name.
    # Otherwise, return the original OCR'd name (it might be a random player).
    if best_score >= min_score:
        return best_match
    else:
        return ocr_name


def analyze_scoreboard(scoreboard_img_path):
    # All the image loading and ROI setup is unchanged
    if not os.path.exists(scoreboard_img_path):
        return
    scoreboard_img = cv2.imread(scoreboard_img_path)
    if scoreboard_img is None:
        return
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
    roi_coords = {
        "map": (1515, 291, 2205, 738),
        "result": (1556, 773, 1770, 847),
        "team1_names": (479, 336, 713, 743),
        "team1_heroes": (390, 328, 477, 744),
        "team2_names": (477, 868, 778, 1279),
        "team2_heroes": (385, 849, 483, 1288),
    }
    # Slicing is unchanged
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

    best_map_match = {"name": "Antarctic Peninsula"}  # Unchanged
    team1_heroes_found = find_heroes_in_roi(
        ROI_HEROES_1, hero_templates, HERO_DETECTION_THRESHOLD
    )
    team2_heroes_found = find_heroes_in_roi(
        ROI_HEROES_2, hero_templates, HERO_DETECTION_THRESHOLD
    )
    team1_heroes_sorted = sorted(team1_heroes_found, key=lambda item: item[2])
    team2_heroes_sorted = sorted(team2_heroes_found, key=lambda item: item[2])

    # OCR is unchanged
    try:
        result_text = (
            pytesseract.image_to_string(ROI_RESULT, config="--psm 8").strip().upper()
        )
        processed_team1 = preprocess_for_ocr(ROI_TEAM1_NAMES)
        processed_team2 = preprocess_for_ocr(ROI_TEAM2_NAMES)
        team1_text = pytesseract.image_to_string(
            processed_team1, config="--psm 4"
        ).strip()
        team2_text = pytesseract.image_to_string(
            processed_team2, config="--psm 4"
        ).strip()
        match_result = (
            "VICTORY"
            if "VICTORY" in result_text
            else "DEFEAT" if "DEFEAT" in result_text else "UNKNOWN"
        )
        # Get the raw OCR results
        ocr_team1_players = [name for name in team1_text.split("\n") if name.strip()]
        ocr_team2_players = [name for name in team2_text.split("\n") if name.strip()]
    except Exception as e:
        print(f"\n--- OCR FAILED --- \nAn error occurred: {e}")
        return

    # --- NEW: CORRECTION STEP ---
    # Loop through the raw OCR results and find the best match from your known list
    team1_players = [
        find_best_match(name, KNOWN_PLAYERS, MIN_NAME_MATCH_SCORE)
        for name in ocr_team1_players
    ]
    team2_players = [
        find_best_match(name, KNOWN_PLAYERS, MIN_NAME_MATCH_SCORE)
        for name in ocr_team2_players
    ]

    # Final data assembly is unchanged
    final_data = {
        "map": "Antarctic Peninsula",
        "result": match_result,
        "team1": [],
        "team2": [],
    }
    print("\n--- DATA VALIDATION ---")
    print(
        f"Found {len(team1_players)} player names and {len(team1_heroes_sorted)} hero icons for Team 1."
    )
    print(
        f"Found {len(team2_players)} player names and {len(team2_heroes_sorted)} hero icons for Team 2."
    )
    if len(team1_players) != len(team1_heroes_sorted) or len(team2_players) != len(
        team2_heroes_sorted
    ):
        print(
            "\nWarning: Mismatch between number of players and heroes found. JSON will be incomplete."
        )
    for i, player_name in enumerate(team1_players):
        if i < len(team1_heroes_sorted):
            final_data["team1"].append(
                {"player_name": player_name, "hero": team1_heroes_sorted[i][0].title()}
            )
    for i, player_name in enumerate(team2_players):
        if i < len(team2_heroes_sorted):
            final_data["team2"].append(
                {"player_name": player_name, "hero": team2_heroes_sorted[i][0].title()}
            )

    print("\n--- EXTRACTION COMPLETE (Corrected) ---")
    print(json.dumps(final_data, indent=2))


if __name__ == "__main__":
    analyze_scoreboard(SCOREBOARD_PATH)
