import logging
from os import PathLike

import ffmpeg

from .error import NoAudioStreamsError, AudioTrimError


logger = logging.getLogger(__name__)


def trim_audio_and_convert_to_mp3(input_file: PathLike, start: float, end: float) -> bytes:
    """
    Convert a segment of an audio or video file to MP3 format and return as bytes.
    The segment is specified by start and end times in seconds.

    Args:
    input_file (str): Path to the input file.
    start (float): Start time of the segment in seconds.
    end (float): End time of the segment in seconds.

    Returns:
    bytes: MP3 data of the specified segment.
    """
    duration = end - start
    input_stream = ffmpeg.input(str(input_file), ss=start, t=duration).audio

    def make_output_stream(stream: ffmpeg.Stream) -> ffmpeg.Stream:
        return stream.output('pipe:', format='mp3', acodec='libmp3lame', ab='128k', map_metadata="-1")

    stream_options = [
        make_output_stream(input_stream),
        make_output_stream(input_stream.filter("aresample", min_hard_comp="0.100000", first_pts="0", **{"async": "1"}))
    ]

    at_least_some_data = b""
    for stream in stream_options:
        try:
            out, err = stream.run(capture_stdout=True, capture_stderr=True)
            if b"nothing was encoded" in err:
                raise AudioTrimError(f"ffmpeg reported that nothing was encoded for trim from {start:.2f} to {end:.2f}")
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
