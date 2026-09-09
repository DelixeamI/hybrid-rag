"""What an index was built from. If any of it changes, the stored vectors and chunks are stale."""

import hashlib
import json

# Bump when the code changes what a given file produces. Forgetting to bump leaves a stale index
# that the loader would keep; the tests pin the fingerprint inputs but cannot see logic changes.
PARSER_VERSION = 2  # 2: implicit part numbers, hyphenated article numbers (341.1-1)
CHUNKER_VERSION = 1


def index_fingerprint(
    source_sha256: str, embedder_name: str, embedding_dim: int, max_tokens: int
) -> str:
    payload = {
        "source": source_sha256,
        "embedder": embedder_name,
        "dim": embedding_dim,
        "max_tokens": max_tokens,
        "parser": PARSER_VERSION,
        "chunker": CHUNKER_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
