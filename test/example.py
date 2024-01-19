import asyncio
import logging
from whisperstream import atranscribe_streaming_simple


async def main():
    language, gen = await atranscribe_streaming_simple("test_audio.mp3", force_punctuation=True)

    print(language.name)

    async for segment in gen:
        print(segment.text, end="", flush=True)


if __name__ == "__main__":
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
