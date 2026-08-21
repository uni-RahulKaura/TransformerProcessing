"""Stage 4 -- calling the summarisation model.

One model: BART-large-CNN. It writes a new sentence rather than quoting one, and its
1,024-token reading limit is why a long section is split into parts and each part summarised
separately.

Nothing is sampled, so a re-run reproduces the same summaries word for word:

  do_sample=False          makes the run reproducible
  min_new_tokens=12        stops it answering in three words
  max_new_tokens=70        (80 when combining part summaries)
  no_repeat_ngram_size=3   stops the repetition loops it falls into on tables
  num_beams=4

WHY THIS MODEL. Four were measured on the corpus this was built for, weighting usefulness
50%, faithfulness 30%, speed 20%:

  BART-large-CNN        7.2/10   5.64 s per section   chosen
  DistilBART-CNN        7.1/10   3.15 s               0.1 behind, 1.8x faster
  MiniLM extractive     6.5/10   0.07 s               safe but scores 3/10 on usefulness
  LongT5-16384          4.0/10   2.37 s               the only one that invented content

Splitting long sections BEAT reading them whole: LongT5 read a 3,089-word section in one pass
and invented a detail that was not in the document, while BART read the same section in five
parts and every part summary was accurate.

KNOWN BEHAVIOUR, measured. BART-large-CNN is fine-tuned on CNN/DailyMail, whose reference
summaries are largely the article's opening paragraphs, so the model learned to copy: about
88% of a summary is verbatim 5-word runs from its source. For an index that is mostly a
safety property -- copied text cannot invent a fact, which is why prose summaries produce no
fabricated figures. Table figures are NOT produced here for the same reason in reverse: BART
does invent numbers (it returned $353.00 for $353,377), so tables are summarised by
arithmetic over their cells in tables.py instead.
"""
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .chunking import SENT, split_parts

# Run on the GPU when there is one. A T4 is sm_75, which does fp16 but not bf16, so fp16 is
# the only half precision available -- and it is the point of going to the GPU at all: BART's
# 400M parameters at fp32 on one CPU thread is what made the full 33-file run take hours.
REPO = "facebook/bart-large-cnn"
LIMIT = 1024

DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
if DEVICE == "cpu":
    torch.set_num_threads(1)


def load(repo=REPO):
    """Load the tokenizer and model onto the GPU when there is one.

    A T4 is sm_75: it does fp16 but not bf16, so fp16 is the only half precision available --
    and it is the point of using the GPU at all, since 400M parameters at fp32 on a single CPU
    thread is what made a full corpus run take hours.
    """
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=DTYPE).to(DEVICE).eval()
    return tok, mdl


def summarise(tok, mdl, repo, limit, text, longer=False):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=limit).to(DEVICE)
    g = mdl.generate(**enc, max_new_tokens=80 if longer else 70, min_new_tokens=12,
                     num_beams=4 if "long-t5" not in repo else 2,
                     do_sample=False, no_repeat_ngram_size=3)
    return tok.decode(g[0], skip_special_tokens=True).strip()


