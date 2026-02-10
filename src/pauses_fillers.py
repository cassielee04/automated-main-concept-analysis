import re
import json
def parse_time(time_str):
    """
    Convert a VTT timestamp from "hh:mm:ss,ms" format to seconds.
    For example, "00:00:09,180" becomes 9.18 seconds.
    """
    hours, minutes, sec_ms = time_str.split(":")
    seconds, ms = sec_ms.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 1000.0


def count_fillers_pauses_json(json_data):
    items = json_data['results']['items']

    filler_words = {'uh', 'um'}
    filler_count = 0
    pause_count = 0
    pause_durations = []

    prev_end = None

    for item in items:
        if item['type'] != 'pronunciation':
            continue

        word = item['alternatives'][0]['content'].lower()
        start = float(item['start_time'])
        end = float(item['end_time'])

        if word in filler_words:
            filler_count += 1

        if prev_end is not None:
            pause = start - prev_end
            if pause > 0.25:
                pause_count += 1
                pause_durations.append(pause)

        prev_end = end
    # Compute average pause duration (if any)
    avg_pause = round(sum(pause_durations) / len(pause_durations), 3) if pause_durations else 0.0

    return filler_count, avg_pause, pause_count




def contains_empty_filler(sentence: str) -> bool:
    """
    Return True if the sentence includes filler words like 'uh' or 'um'.
    """
    return bool(re.search(r'\b(uh|um)\b', sentence, re.IGNORECASE))


# Example usage:
# if __name__ == '__main__':
#     file_path = "../hyperscanning/data/dyad3_SLP_recording_Independent_Reading_transcribed.json"
#     # Read and parse the JSON file
#     try:
#         with open(file_path, 'r') as file:
#             json_data = json.load(file)  # Parse the JSON file into a dictionary
#         fillers, avg_pause, pause_count = count_fillers_pauses_json(json_data)
#         print(f"Number of fillers (um/uh variants): {fillers}")
#         print(f"Average pause duration (seconds, gaps > 0.2 s): {avg_pause:.2f}")
#         print(f"Number of pauses: {pause_count}")
#     except FileNotFoundError:
#         print(f"Error: The file {file_path} was not found.")
#     except json.JSONDecodeError:
#         print(f"Error: The file {file_path} contains invalid JSON.")
#     except KeyError as e:
#         print(f"Error: Missing key in JSON data: {e}")
