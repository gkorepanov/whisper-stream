# Whisper API Streaming

OpenAI has recently [published](https://platform.openai.com/docs/api-reference/audio/create) their Whisper model API for audio transcription.

Unfortunately, this API does not provide streaming capabilities. This project aims to provide a streaming interface to the OpenAI API.


## Functionality
Currently only streaming of response is supported. If you need also streaming of input aduio, please open an issue and describe what you need, it should be easy to implement.


## Installation

```bash
pip install git+https://github.com/gkorepanov/whisper-stream.git
```


## CLI usage
To transcribe a file, run the following command:
```bash
OPENAI_API_KEY=<KEY> whisperstream /path/to/your/audio/file.ogg -l en
```
You can omit the language parameter, it will be detected automatically.


## Usage

```python
from whisperstream import atranscribe_streaming_simple
path = '/path/to/your/audio/file.ogg'
language, gen = await atranscribe_streaming_simple(path)

# language is a Lang object from [iso369 lib](https://github.com/LBeaudoux/iso639)
print(language.name)

async for segment in gen:
    # segment is an OpenAI Python API object
    # it has `start`, `end`, `text` attributes
    print(segment.text, end="")
```
