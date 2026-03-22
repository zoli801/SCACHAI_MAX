import os
import json
import requests
from pathlib import Path
from datetime import datetime

BOT_TOKEN = ""
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
FILE_BASE_URL = f"https://api.telegram.org/file/bot{BOT_TOKEN}"

DATA_DIR = Path("data")
OFFSET_FILE = Path("last_update_id.txt")

ALLOWED_EXTENSIONS = {".py", ".ipynb"}


def load_offset() -> int:
    if OFFSET_FILE.exists():
        try:
            return int(OFFSET_FILE.read_text(encoding="utf-8").strip())
        except:
            return 0
    return 0


def save_offset(offset: int):
    OFFSET_FILE.write_text(str(offset), encoding="utf-8")


def get_updates(offset: int):
    resp = requests.get(
        f"{BASE_URL}/getUpdates",
        params={
            "offset": offset,
            "timeout": 0,
            "allowed_updates": ["message"]
        },
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def get_file_info(file_id: str):
    resp = requests.get(
        f"{BASE_URL}/getFile",
        params={"file_id": file_id},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError("Не удалось получить file_path")
    return data["result"]


def download_file(file_path: str, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(f"{FILE_BASE_URL}/{file_path}", timeout=60)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)


def get_user_dir(user_id: int) -> Path:
    user_dir = DATA_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def append_text(user_id: int, text: str, msg_dt: str):
    user_dir = get_user_dir(user_id)
    text_file = user_dir / "text.txt"

    with open(text_file, "a", encoding="utf-8") as f:
        f.write(f"[{msg_dt}] {text}\n")


def unique_file_path(folder: Path, filename: str) -> Path:
    path = folder / filename
    if not path.exists():
        return path

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 1
    while True:
        new_path = folder / f"{stem}_{i}{suffix}"
        if not new_path.exists():
            return new_path
        i += 1


def handle_document(message: dict, user_id: int):
    document = message.get("document")
    if not document:
        return "skip"

    filename = document.get("file_name")
    file_id = document.get("file_id")

    if not filename or not file_id:
        return "skip"

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return "skip"

    user_dir = get_user_dir(user_id)
    save_path = unique_file_path(user_dir, filename)

    file_info = get_file_info(file_id)
    file_path = file_info["file_path"]

    download_file(file_path, save_path)
    return f"saved file: {save_path}"


def handle_text(message: dict, user_id: int):
    text = message.get("text")
    if not text:
        return "skip"

    msg_dt = datetime.fromtimestamp(message["date"]).strftime("%Y-%m-%d %H:%M:%S")
    append_text(user_id, text, msg_dt)
    return "saved text"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    offset = load_offset()
    data = get_updates(offset)

    if not data.get("ok"):
        print("Ошибка Telegram API")
        return

    updates = data.get("result", [])
    if not updates:
        print("Новых сообщений нет")
        return

    max_update_id = offset

    for upd in updates:
        max_update_id = max(max_update_id, upd["update_id"] + 1)

        message = upd.get("message")
        if not message:
            continue

        user = message.get("from", {})
        user_id = user.get("id")
        if not user_id:
            continue

        result = "skip"

        if "text" in message:
            result = handle_text(message, user_id)

        elif "document" in message:
            result = handle_document(message, user_id)

        print(f"user_id={user_id} -> {result}")

    save_offset(max_update_id)
    print("Готово")


if __name__ == "__main__":
    main()
