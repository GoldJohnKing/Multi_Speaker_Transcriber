"""Shared constants for the transcription pipeline."""

# Sentence-ending punctuation — hard split points, discarded from output.
SENTENCE_END = frozenset("。！？!?…—")

# Clause-internal punctuation — soft split points (duration-gated), discarded from output.
CLAUSE_END = frozenset("，；：,;:")

# Punctuation characters that may appear in word-level ASR output but lack
# independent timestamps. Used for speaker attribution alignment and
# character-level timestamp handling.
ALIGNMENT_PUNCT = frozenset("，。！？,;:、；：…—!?·")
