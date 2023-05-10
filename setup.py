from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="whisperstream",
    version="0.1.0",
    description="Thin wrapper around OpenAI Whisper API enabling streaming transcription",
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    url="https://github.com/gkorepanov/whisperstream",
    author="George Korepanov",
    author_email="gkorepanov.gk@gmail.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "openai",
        "iso639-lang",
        "pydub",
        "click",
    ],
    entry_points={
        'console_scripts': [
            'transcribe = whisperstream.cmd:transcribe',
        ],
    },
)
