import os

def getenv(key: str, default=0): return type(default)(os.getenv(key, default))

SPEAKER_EMBD = getenv('SPEAKER_EMBD', 0)
