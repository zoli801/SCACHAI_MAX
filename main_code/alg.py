import json
from pathlib import Path


FOLDER_PATH = Path("/Users/dmitrii/PycharmProjects/heroes/main_code/cheked_code")
OUTPUT_JSON = Path("/Users/dmitrii/PycharmProjects/heroes/main_code/lb/db.json")
DEFAULT_NICK = "unknown"


def count_chars(file_path: Path) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        return len(f.read())


def build_leaderboard(folder_path: Path) -> dict:
    files_data = []

    for file_path in folder_path.iterdir():
        if file_path.is_file():
            try:
                length = count_chars(file_path)
                files_data.append((file_path.name, length))
            except Exception as e:
                print(f"Ошибка при чтении {file_path.name}: {e}")

    files_data.sort(key=lambda x: x[1])  # чем меньше длина, тем выше

    leaderboard = {}
    for i, (file_name, length) in enumerate(files_data, start=1):
        leaderboard[str(i)] = [DEFAULT_NICK, length, file_name]

    return leaderboard


def main():
    leaderboard = build_leaderboard(FOLDER_PATH)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    print(f"Готово: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()