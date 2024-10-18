from typing import Optional, Literal

import logging
from os import PathLike
import re

import ffmpeg

from .error import NoAudioStreamsError, AudioTrimError


logger = logging.getLogger(__name__)


def trim_audio_and_convert(
    input_file: PathLike,
    output_file: Optional[PathLike] = None,
    start: float = 0.0,
    end: Optional[float] = None,
    audio_format: Literal["wav", "mp3"] = "wav",
) -> bytes:
    """
    Convert a segment of an audio or video file to MP3 format and return as bytes.
    The segment is specified by start and end times in seconds.

    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file, if None, will return bytes.
        start (float): Start time of the segment in seconds.
        end (float): End time of the segment in seconds.
        audio_format (Literal["wav", "mp3"]): Audio format to convert to.

    Returns:
        bytes: MP3 data of the specified segment.
    """
    assert start >= 0, f"Start time must be non-negative, got {start}"
    kwargs = {}
    if end is not None:
        duration = end - start
        assert duration > 0, f"Duration must be positive, got {duration}"
        kwargs["t"] = duration
    if start > 0:
        kwargs["ss"] = start
    input_stream = ffmpeg.input(str(input_file), **kwargs).audio

    def make_output_stream(stream: ffmpeg.Stream, output_file: Optional[PathLike]) -> ffmpeg.Stream:
        if audio_format == "wav":
            kwargs = {"format": "wav", "ar": "16000", "ac": "1", "map_metadata": "-1"}
        elif audio_format == "mp3":
            kwargs = {"format": "mp3", "acodec": "libmp3lame", "ab": "128k", "map_metadata": "-1"}
        else:
            raise ValueError(f"Unknown audio format: {audio_format}")
        if output_file is not None:
            return stream.output(str(output_file), **kwargs)
        else:
            return stream.output('pipe:', **kwargs)

    stream_options = [
        make_output_stream(input_stream, output_file),
        make_output_stream(input_stream.filter("aresample", min_hard_comp="0.100000", first_pts="0", **{"async": "1"}), output_file),
    ]

    at_least_some_data = b""
    for stream in stream_options:
        try:
            out, err = stream.run(capture_stdout=True, capture_stderr=True)
            if b"nothing was encoded" in err:
                raise AudioTrimError(f"ffmpeg reported that nothing was encoded for trim from {start} to {end}")
            return out
        except ffmpeg.Error as e:
            out = e.stdout
            cmd = " ".join(stream.compile())
            logger.warning(
                f"ffmpeg trim error for command: {cmd}, "
                f"got {len(out)} bytes, ffmpeg stderr:\n{e.stderr.decode()}"
            )
            if len(out) > len(at_least_some_data):
                at_least_some_data = out
            continue
    else:
        if len(at_least_some_data) > 0:
            logger.warning(f"All ffmpeg trim attempts failed, returning partial data, {len(at_least_some_data)} bytes")
            return at_least_some_data
        else:
            raise RuntimeError(f"All ffmpeg trim attempts failed, got empty output from ffmpeg")


def get_audio_duration(file_path: PathLike) -> float:
    try:
        return get_audio_duration_ffprobe(file_path)
    except Exception as e:
        logger.warning(f"Error getting audio duration using ffprobe: {e}")
        return get_audio_duration_decode(file_path)


def get_audio_duration_ffprobe(file_path: PathLike) -> float:
    try:
        metadata = ffmpeg.probe(str(file_path))
    except ffmpeg.Error as e:
        raise RuntimeError(f"Could not get duration using ffprobe, ffmpeg stderr:\n{e.stderr.decode()}")

    audio_streams = [s for s in metadata["streams"] if s["codec_type"] == "audio"]
    if len(audio_streams) == 0:
        raise NoAudioStreamsError("No audio streams found")

    audio_streams = [s for s in audio_streams if "duration" in s]
    if len(audio_streams) == 0:
        logger.warning("No audio streams with duration found")
    else:
        return max(float(s["duration"]) for s in audio_streams)

    try:
        return float(metadata["format"]["duration"])
    except Exception as e:
        raise RuntimeError(f"Could not get duration using ffprobe") from e

def get_audio_duration_decode(file_path: PathLike) -> float:
    stream = ffmpeg.input(str(file_path))
    output = stream.output('pipe:', format='null', vn=None, threads=1)
    try:
        out, err = output.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        err = e.stderr
    time_matches = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", err.decode())

    if time_matches:
        last_time_match = time_matches[-1]
        hours, minutes, seconds = last_time_match
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise RuntimeError(f"Could not get duration using ffmpeg decode, ffmpeg stderr:\n{err.decode()}")
