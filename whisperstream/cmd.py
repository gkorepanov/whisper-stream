import click
import asyncio
import os

from whisperstream import atranscribe_streaming_simple


async def _run(path: os.PathLike, language_code: str = None):
    lang, segments =  await atranscribe_streaming_simple(path, language=language_code)
    if language_code is None:
        click.echo(f"Detected language: {lang.name}")
    async for segment in segments:
        click.echo(segment.text, nl=False)
    click.echo("")


@click.command()
@click.argument('path', type=str)
@click.option('--language-code', '-l', type=str, default=None)
def transcribe(path: os.PathLike, language_code: str = None):
    """Transcribe audio file and print transcribed text to console
    """
    asyncio.run(_run(path, language_code=language_code))
