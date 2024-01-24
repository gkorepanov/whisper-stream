import asyncio
import logging
from whisperstream import atranscribe_streaming_simple


async def main():
    language, gen = await atranscribe_streaming_simple("test_audio.mp3", force_punctuation=True)

    print(language.name)
    result = ""

    async for segment in gen:
        print(segment.text, end="", flush=True)
        result += segment.text

    print("\nResult:", result)


if __name__ == "__main__":
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
