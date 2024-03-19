from typing import Tuple, AsyncIterator, Callable, BinaryIO, Optional
from pathlib import Path
import os
import logging

from io import BytesIO
from iso639 import Lang
from functools import partial

import asyncio
from concurrent.futures import Executor

import openai
from openai import AsyncOpenAI
from openai.types.audio import Transcription

from whisperstream.languages import (
    get_punctuation_prompt_for_lang,
    get_lang_from_name,
    SUPPORTED_LANGUAGES,
)
from whisperstream.error import UnsupportedLanguageError
from whisperstream.trim import get_audio_duration, trim_audio_and_convert_to_mp3


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


def is_punctuation_present(text: str) -> bool:
    if not any(i.isupper() for i in text):
        return False
    if not any(i in "!(),.:;?" for i in text):
        return False
    return True


async def default_atranscribe_fn(
    model: str,
    file: BinaryIO,
    duration_seconds: float,
    *args,
    **kwargs,
) -> Transcription:
    if "prompt" in kwargs:
        kwargs["prompt"] = kwargs["prompt"][-1000:]  # limit prompt to last 1000 chars, which is > OpenAI limit
    api_key = kwargs.pop("api_key", openai.api_key)
    client = AsyncOpenAI(api_key=api_key)
    return await client.audio.transcriptions.create(
        model=model,
        file=file,
        *args,
        **kwargs,
    )


async def atranscribe_streaming_simple(
    path: os.PathLike,
    model: str = 'whisper-1',
    chunk_size_fn: Callable[[int], int] = default_chunk_size_fn,
    atranscribe_fn: Callable[..., Transcription] = default_atranscribe_fn,
    language: Optional[Lang] = None,
    executor: Optional[Executor] = None,
    force_punctuation: bool = False,
    **kwargs,
) -> Tuple[Lang, AsyncIterator[Transcription]]:
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
        force_punctuation: (bool, optional): Locates rare cases of missed punctuation
            and forces it if necessary
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
        force_punctuation=force_punctuation,
        **kwargs,
    )
    it = gen.__aiter__()
    first_elem = await it.__anext__()

    # remove leading spaces from first segment
    if len(first_elem.segments) > 0:
        first_elem.segments[0]["text"] = first_elem.segments[0]["text"].lstrip()

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
    atranscribe_fn: Callable[..., Transcription] = default_atranscribe_fn,
    language: Optional[Lang] = None,
    executor: Optional[Executor] = None,
    force_punctuation: bool = False,
    **kwargs,
) -> AsyncIterator[Transcription]:
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
        force_punctuation: (bool, optional): Locates rare cases of missed punctuation
            and forces it if necessary.
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

    # in order to avoid conflicts you cannot use both prompt and force_punctuation
    if force_punctuation and kwargs.get("prompt") is not None:
        raise ValueError(
            "Cannot enforce punctuation when custom prompt is used. "
            "Please set `force_punctuation` to False"
        )

    path = Path(path).resolve()

    def _f():
        return get_audio_duration(path)
    if executor is not None:
        loop = asyncio.get_running_loop()
        audio_duration = await loop.run_in_executor(executor, _f)
    else:
        audio_duration = _f()

    __transcribe = partial(
        _transcribe,
        path=path,
        executor=executor,
        atranscribe_fn=atranscribe_fn,
        model=model,
        **kwargs,
    )

    logger.debug(f"Audio duration: {audio_duration}")

    def _get_end(start: int, chunk_index: int) -> int:
        chunk_size = chunk_size_fn(chunk_index)
        end = start + chunk_size
        if (audio_duration - end) < chunk_size:  # do not make the last chunk too small
            end = audio_duration
        return end

    start = 0
    chunk_index = 0
    end = _get_end(start, chunk_index)

    if force_punctuation and language is not None:
        kwargs["prompt"] = get_punctuation_prompt_for_lang(language)

    r = await __transcribe(start=start, end=end, **kwargs)

    if language is None:
        language = get_lang_from_name(r.language)
        kwargs['language'] = language.pt1

    if force_punctuation:
        if not is_punctuation_present(r.text):
            logger.info(
                "Punctuation is not present, adding "
                "prefix to force punctuation and restarting transcription"
            )
            kwargs["prompt"] = get_punctuation_prompt_for_lang(language) + " "
            r = await __transcribe(start=start, end=end, **kwargs)
        else:  # at least capitalize the first letter
            def _capitalize(text):
                text = text.lstrip()
                return text[:1].upper() + text[1:]
            r.text = _capitalize(r.text)
            if len(r.segments) > 0:
                r.segments[0].text = _capitalize(r.segments[0].text)

    while True:
        # update seek and start/end times in all segments
        for segment in r.segments:
            segment["seek"] += start
            segment["start"] += start
            segment["end"] += start

        # if we are at the end of the audio, return all segments
        # returned by OpenAI
        if end >= audio_duration:
            yield r
            logger.debug(f"End of audio, yield text: {r.text}")
            return

        logger.debug(f"{len(r.segments)} segments returned for start = {start} end = {end}:")
        for segment in r.segments:
            logger.debug(f"\ttext: {segment['text']}")
            logger.debug(f"\tstart: {segment['start']}")
            logger.debug(f"\tend: {segment['end']}")

        # if only one segment was returned, we have to continue
        # transcription from the end of the segment
        if len(r.segments) == 0:
            start = end
        elif len(r.segments) == 1:
            start = end
        else:
            # if more than one segment was returned, we discard the last incomplete one
            # and continue transcription from the end of the second to last
            max_segments_to_skip = max(
                1, len(r.segments) - 1
            )
            for _i in range(max_segments_to_skip):
                logger.debug(f"Skipping segment -{_i}")
                r.segments = r.segments[:-1]
                if (r.segments[-1]["end"] < end):
                    break
                else:
                    logger.debug(f"Strange segment end {r.segments[-1]['end']}, skipping it. All segments:")
            else:
                logger.warning(
                    f"Segment end {r.segments[-1]['end']} is greater than chunk end {end} even after "
                    f"discarding {max_segments_to_skip} segments"
                )
            start = min(r.segments[-1]["end"], end)
            r.text = ''.join(x.text for x in r.segments)

        # update prompt with the text returned by OpenAI for the previous chunk
        # to produce coherent transcription
        kwargs["prompt"] = kwargs.get("prompt", "") + r.text

        yield r
        logger.debug(f"Yield text: {r.text}")

        chunk_index += 1
        end = _get_end(start, chunk_index)
        r = await __transcribe(start=start, end=end, **kwargs)


async def _transcribe(
    *,
    path: os.PathLike,
    start: int,
    end: int,
    executor: Optional[Executor],
    atranscribe_fn: Callable[..., Transcription],
    model: str,
    **kwargs,
) -> Transcription:
    logger.debug("Crop audio")

    def _f():
        return trim_audio_and_convert_to_mp3(path, start, end)

    if executor is not None:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, _f)
    else:
        data = _f()

    # for debugging
    # with open(f"debug_{start:.3f}_{end:.3f}.mp3", "wb") as f:
    #     f.write(data)

    f = BytesIO(data)
    f.name = "audio.mp3"

    logger.debug(f"Transcribe request with start = {start} end = {end}")
    r = await atranscribe_fn(
        model=model,
        file=f,
        duration_seconds=(end - start),
        **kwargs,
    )
    logger.debug("Transcribe request finished")
    return r
