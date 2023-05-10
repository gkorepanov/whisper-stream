from typing import Tuple, AsyncIterator
import os
from io import BytesIO
import logging
from iso639 import Lang
import openai
from openai.openai_object import OpenAIObject
from pathlib import Path
import pydub


logger = logging.getLogger(__name__)


OPENAI_WHISPER_MODEL_CHUNK_SIZE_SECONDS = 30


async def atranscribe_streaming_simple(
    path: os.PathLike,
    model: str = 'whisper-1',
    **kwargs,
) -> Tuple[Lang, AsyncIterator[OpenAIObject]]:
    """High level wrapper for streaming tracription with simple interface.

    Args:
        path (os.PathLike): Path to audio file
        model (str, optional): OpenAI model name. Defaults to 'whisper-1'.
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
    gen = atranscribe_streaming(path, model, **kwargs)
    it = gen.__aiter__()
    first_elem = await it.__anext__()
    first_elem.segments[0].text = first_elem.segments[0].text.lstrip()
    async def _gen():
        for segment in first_elem.segments:
            yield segment
        async for elem in it:
            for segment in elem.segments: 
                yield segment
    return Lang(first_elem.language.capitalize()), _gen()


async def atranscribe_streaming(
    path: os.PathLike,
    model: str = 'whisper-1',
    **kwargs,
) -> AsyncIterator[OpenAIObject]:
    """Low level OpenAI Whisper API wrapper for streaming transcription.

    Args:
        path (os.PathLike): Path to audio file
        model (str, optional): OpenAI model name. Defaults to 'whisper-1'.
        kwargs: Additional arguments for OpenAI API
    
    Returns:
        AsyncGenerator[OpenAIObject]: Generator of OpenAI responses for consecutive
            chunks of audio
    """

    # force "verbose_json" response format to get segments
    kwargs["response_format"] = "verbose_json"

    path = Path(path)
    audio = pydub.AudioSegment.from_file(str(path))

    # the whisper model inference uses 30 seconds of audio at a time
    # so using chunks of smaller size is useless because it will be padded
    chunk_size = OPENAI_WHISPER_MODEL_CHUNK_SIZE_SECONDS * 1000

    async def _transcribe(start: int):
        logger.debug("Crop audio")
        segment = audio[start:start+chunk_size]
        f = BytesIO()
        f.name = "voice.mp3"
        segment.export(f, format="mp3")
        f.seek(0)
        logger.debug(f"Transcribe request with start = {start}")
        r = await openai.Audio.atranscribe(
            model,
            f,
            **kwargs,
        )
        logger.debug("Transcribe request finished")
        return r


    total_len = len(audio)
    logger.debug(f"Audio len: {total_len}")
    start = 0
    r = await _transcribe(start)
    kwargs['language'] = Lang(r.language.capitalize()).pt1
    
    while True:
        # update seek and start/end times in all segments
        for segment in r.segments:
            segment.seek += start / 1000
            segment.start += start / 1000
            segment.end += start / 1000

        # if we are at the end of the audio, return all segments
        # returned by OpenAI
        if start + chunk_size >= total_len:
            yield r
            return

        # if only one segment was returned, we have to continue
        # transcription from the end of the segment
        if len(r.segments) == 1:
            start += chunk_size
        else:  
            # if more than one segment was returned, we discard the last incomplete one
            # and continue transcription from the end of the second to last
            r.segments = r.segments[:-1]
            assert r.segments[-1].end < (start + chunk_size / 1000)
            start = int(r.segments[-1].end * 1000)
            r.text = ''.join(x.text for x in r.segments)

        # update prompt with the text returned by OpenAI for the previous chunk
        # to produce coherent transcription
        kwargs["prompt"] = kwargs.get("prompt", "") + r.text

        yield r

        r = await _transcribe(start)
