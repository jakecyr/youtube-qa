from typing import NamedTuple


class VideoInfo(NamedTuple):
    title: str
    id: str
    transcript: str
    url: str
    views: int
    date: str
    thumbnails: list[str]


class VideoSource(NamedTuple):
    title: str
    id: str
    url: str
    thumbnails: list[str]
