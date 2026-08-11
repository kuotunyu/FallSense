"""以有 timeout 的匿名 HTTP listing 走訪公開 Google Drive folder。"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FOLDER_MIME = "application/vnd.google-apps.folder"
FOLDER_URL = "https://drive.google.com/drive/folders/{folder_id}?hl=en"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
)


@dataclass(frozen=True, order=True)
class DriveEntry:
    """單一 Google Drive file；folder 不會出現在最終 entries。"""

    remote_path: str
    file_id: str
    mime_type: str


@dataclass(frozen=True)
class DriveListing:
    entries: tuple[DriveEntry, ...]
    folders_visited: int


@dataclass(frozen=True)
class _Folder:
    folder_id: str
    remote_path: str


ParsedItem = tuple[str, str, str]
PageParser = Callable[[str, str], tuple[object, list[ParsedItem]]]


def _gdown_page_parser() -> PageParser:
    """延用 gdown 維護的 Drive HTML parser，但自行加 timeout 與平行走訪。"""

    module = importlib.import_module("gdown.download_folder")
    return cast(PageParser, module._parse_google_drive_file)


class GoogleDriveFolderLister:
    """以 breadth-first、每層平行請求列出 folder；不下載任何 file content。"""

    def __init__(
        self,
        *,
        workers: int = 8,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        parser: PageParser | None = None,
    ) -> None:
        if workers < 1 or retries < 1:
            raise ValueError("workers 與 retries 必須至少為 1")
        self.workers = workers
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.parser = parser or _gdown_page_parser()

    def _fetch(self, folder: _Folder) -> tuple[list[_Folder], list[DriveEntry]]:
        url = FOLDER_URL.format(folder_id=folder.folder_id)
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                request = Request(url, headers={"User-Agent": USER_AGENT})
                with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                    html = response.read().decode("utf-8")
                    final_url = response.url
                _, items = self.parser(final_url, html)
                folders: list[_Folder] = []
                entries: list[DriveEntry] = []
                for item_id, item_name, mime_type in items:
                    remote_path = "/".join(part for part in (folder.remote_path, item_name) if part)
                    if mime_type == FOLDER_MIME:
                        folders.append(_Folder(item_id, remote_path))
                    else:
                        entries.append(DriveEntry(remote_path, item_id, mime_type))
                return folders, entries
            except (HTTPError, URLError, TimeoutError, RuntimeError, UnicodeError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(
            f"無法列出 Drive folder {folder.remote_path or folder.folder_id}"
        ) from last_error

    def list(self, folder_id: str) -> DriveListing:
        """完整走訪 folder；任一子目錄失敗就整體失敗，不產生部分 manifest。"""

        current = [_Folder(folder_id, "")]
        entries: list[DriveEntry] = []
        folders_visited = 0
        while current:
            with ThreadPoolExecutor(max_workers=min(self.workers, len(current))) as executor:
                pages = list(executor.map(self._fetch, current))
            folders_visited += len(current)
            current = []
            for child_folders, child_entries in pages:
                current.extend(child_folders)
                entries.extend(child_entries)
        return DriveListing(tuple(sorted(entries)), folders_visited)
