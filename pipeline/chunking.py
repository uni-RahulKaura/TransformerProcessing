"""Stage 3 -- splitting a section the model cannot read in one pass.

Every summarisation model has a hard input limit. Left alone it reads up to that limit and
SILENTLY DISCARDS the rest -- no error, no warning, no way to tell from the output that it
happened. On the document this was built for, seven sections were over the limit and the
largest was 3,089 words, so the default behaviour would have thrown away most of them.

So: measure the section in the model's own tokeniser. If it fits, leave it alone. If it
does not, accumulate whole sentences into a part until the next sentence would push it past
88% of the limit, then start a new part. Each part is summarised on its own and the section
summary is built from the parts.

The 88% is headroom, so a long final sentence cannot overshoot. Sentences are never cut in
half. Nothing goes unread.
"""
import re

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_parts(tok, text, limit):
    if len(tok(text).input_ids) <= limit:
        return [text]
    piece = int(limit * 0.88)
    chunks, cur = [], ""
    for sent in SENT.split(text):
        trial = (cur + " " + sent).strip()
        if len(tok(trial).input_ids) > piece and cur:
            chunks.append(cur)
            cur = sent
        else:
            cur = trial
    if cur:
        chunks.append(cur)
    return chunks or [text]
