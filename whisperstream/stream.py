from typing import Tuple, AsyncIterator, Callable, BinaryIO, Optional
import os
from io import BytesIO
import logging
from iso639 import Lang
import openai
from openai.openai_object import OpenAIObject
from pathlib import Path
import pydub
import asyncio
from concurrent.futures import Executor

from whisperstream.languages import (
    get_lang_from_name,
    get_lang_name,
    SUPPORTED_LANGUAGES,
)
from whisperstream.error import UnsupportedLanguageError


logger = logging.getLogger(__name__)


# the whisper model inference uses 30 seconds of audio at a time
# so using chunks of smaller size is useless because they will be padded
OPENAI_WHISPER_MODEL_CHUNK_SIZE_SECONDS = 30



def default_chunk_size_fn(index: int) -> int:
    """Convert chunk index to chunk size in seconds.
    
    In the simplest case, the chunk size is constantly 30 seconds, but it can be
    increased for each chunk to reduce the number of requests to
    the OpenAI API.
    """
    factor = 1 if index < 2 else index  # speed up first chunks
    return OPENAI_WHISPER_MODEL_CHUNK_SIZE_SECONDS * factor


async def default_atranscribe_fn(
    model: str,
    file: BinaryIO,
    duration_seconds: float,
    *args,
    **kwargs,
):
    return await openai.Audio.atranscribe(
        model=model,
        file=file,
        *args,
        **kwargs,
    )


async def atranscribe_streaming_simple(
    path: os.PathLike,
    model: str = 'whisper-1',
    chunk_size_fn: Callable[[int], int] = default_chunk_size_fn,
    atranscribe_fn: Callable[..., OpenAIObject] = default_atranscribe_fn,
    language: Optional[Lang] = None,
    executor: Optional[Executor] = None,
    **kwargs,
) -> Tuple[Lang, AsyncIterator[OpenAIObject]]:
    """High level wrapper for streaming tracription with simple interface.

    Args:
        path (os.PathLike): Path to audio file
        model (str, optional): OpenAI model name. Defaults to 'whisper-1'.
        chunk_size_fn (Callable[[int], int], optional): Determines
            chunk size in seconds for each chunk. For fastest streaming, it should
            return a constant value, but it can be increased for each chunk to
            reduce the number of requests to the OpenAI API and improve
            overall latency. Defaults to `default_chunk_size_fn`.
        atranscribe_fn (Callable[..., OpenAIObject], optional): function
            to use for transcription. Defaults to `default_atranscribe_fn`.
            You can pass a custom function to use a different API or add
            custom retry logic.
        language (Optional[Lang], optional): Language of the audio. If not
            specified, it will be detected automatically. Defaults to None.
        executor: (Optional[Executor], optional): Executor used to run blocking code.
        kwargs: Additional arguments for OpenAI API

    Returns:
        Lang, AsyncGenerator[OpenAIObject]: Detected language and generator
            of segments

    Usage:
        >>> lang, segments = await atranscribe_streaming_simple("path/to/audio.mp3")
        >>> print(lang.name)
        >>> async for segment in segments:
        >>>     print(segment.text)
    """
    gen = atranscribe_streaming(
        path=path,
        model=model,
        chunk_size_fn=chunk_size_fn,
        atranscribe_fn=atranscribe_fn,
        language=language,
        executor=executor,
        **kwargs,
    )
    it = gen.__aiter__()
    first_elem = await it.__anext__()

    # remove leading spaces from first segment
    if len(first_elem.segments) > 0:
        first_elem.segments[0].text = first_elem.segments[0].text.lstrip()

    async def _gen():
        for segment in first_elem.segments:
            yield segment
        async for elem in it:
            for segment in elem.segments: 
                yield segment

    return get_lang_from_name(first_elem.language), _gen()


async def atranscribe_streaming(
    path: os.PathLike,
    model: str = 'whisper-1',
    chunk_size_fn: Callable[[int], int] = default_chunk_size_fn,
    atranscribe_fn: Callable[..., OpenAIObject] = default_atranscribe_fn,
    language: Optional[Lang] = None,
    executor: Optional[Executor] = None,
    **kwargs,
) -> AsyncIterator[OpenAIObject]:
    """Low level OpenAI Whisper API wrapper for streaming transcription.

    Args:
        path (os.PathLike): Path to audio file
        model (str, optional): OpenAI model name. Defaults to 'whisper-1'.
        chunk_size_fn (Callable[[int], int], optional): Determines
            chunk size in seconds for each chunk. For fastest streaming, it should
            return a constant value, but it can be increased for each chunk to
            reduce the number of requests to the OpenAI API and improve
            overall latency. Defaults to `default_chunk_size_fn`.
        atranscribe_fn (Callable[..., OpenAIObject], optional): function
            to use for transcription. Defaults to `default_atranscribe_fn`.
            You can pass a custom function to use a different API or add
            custom retry logic.
        language (Optional[Lang], optional): Language of the audio. If not
            specified, it will be detected automatically. Defaults to None.
        executor: (Optional[Executor], optional): Executor used to run blocking code.
        kwargs: Additional arguments for OpenAI API
    
    Returns:
        AsyncGenerator[OpenAIObject]: Generator of OpenAI responses for consecutive
            chunks of audio
    """

    # force "verbose_json" response format to get segments
    kwargs["response_format"] = "verbose_json"

    if language is not None:
        if language not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(f"Language {language} is not supported")
        kwargs["language"] = language.pt1

    path = Path(path)

    def _f():
        return pydub.AudioSegment.from_file(str(path))
    if executor is not None:
        loop = asyncio.get_running_loop()
        audio = await loop.run_in_executor(executor, _f())
    else:
        audio = _f()

    async def _transcribe(start: int, end: int):
        logger.debug("Crop audio")

        def _f():
            f = BytesIO()
            f.name = "voice.mp3"
            segment = audio[start:end]
            segment.export(f, format="mp3")
            f.seek(0)
            return f

        if executor is not None:
            loop = asyncio.get_running_loop()
            f = await loop.run_in_executor(executor, _f())
        else:
            f = _f()

        logger.debug(f"Transcribe request with start = {start} end = {end}")
        r = await atranscribe_fn(
            model=model,
            file=f,
            duration_seconds=(end - start) / 1000,
            **kwargs,
        )
        logger.debug("Transcribe request finished")
        return r

    total_len = len(audio)
    logger.debug(f"Audio len: {total_len}")
    start = 0
    chunk_index = 0
    end = start + int(chunk_size_fn(chunk_index) * 1000)
    r = await _transcribe(start, end)
    if language is None:
        kwargs['language'] = get_lang_from_name(r.language).pt1
    
    while True:
        # update seek and start/end times in all segments
        for segment in r.segments:
            segment.seek += start / 1000
            segment.start += start / 1000
            segment.end += start / 1000

        # if we are at the end of the audio, return all segments
        # returned by OpenAI
        if end >= total_len:
            yield r
            logger.debug(f"End of audio, yield text: {r.text}")
            return

        # if only one segment was returned, we have to continue
        # transcription from the end of the segment
        if len(r.segments) == 0:
            logger.debug(f"No segments returned for start = {start} end = {end}")
            start = end
        elif len(r.segments) == 1:
            logger.debug(f"1 segment returned for start = {start} end = {end}")
            start = end
        else:
            logger.debug(f"{len(r.segments)} segments returned for start = {start} end = {end}")
            # if more than one segment was returned, we discard the last incomplete one
            # and continue transcription from the end of the second to last
            max_segments_to_skip = max(
                1, len(r.segments) - 1
            )
            for _ in range(max_segments_to_skip):
                logger.debug("Skipping segment")
                r.segments = r.segments[:-1]
                if (r.segments[-1].end < (end / 1000)):
                    break
                else:
                    logger.debug("Strange segment end, skipping another segment")
                    logger.debug(f"Segment end: {r.segments[-1].end}")
                    for segment in r.segments:
                        logger.debug(f"text: {segment.text}")
                        logger.debug(f"start: {start}")
                        logger.debug(f"end: {end}")
                        logger.debug(f"{segment}")
            else:
                raise ValueError(f"Segment end is greater than chunk end even after discarding {max_segments_to_skip} segments")
            start = int(r.segments[-1].end * 1000)
            r.text = ''.join(x.text for x in r.segments)

        # update prompt with the text returned by OpenAI for the previous chunk
        # to produce coherent transcription
        kwargs["prompt"] = kwargs.get("prompt", "") + r.text

        yield r
        logger.debug(f"Yield text: {r.text}")

        chunk_index += 1
        end = start + int(chunk_size_fn(chunk_index) * 1000)
        r = await _transcribe(start, end)
