"""Abstractive prose, verbatim figures.

THE REQUIREMENT. Summaries that reword rather than quote -- word-for-word extracts are not
summaries -- but which never misstate a figure, a date or a party name.

WHY THE OBVIOUS APPROACH FAILS. transformers offers `encoder_no_repeat_ngram_size`, which forbids
the decoder from reproducing any n-gram of the input. It does make BART abstractive: measured over
35 sections of five document types, the mean longest-verbatim-run fell from 0.696 of the summary to
0.264 at n=6 and 0.168 at n=4.

It also makes it wrong, because it blocks the input's wording INDISCRIMINATELY -- including the
numbers and names that must be copied exactly. Three independent blind judges, reading only the
source text and told nothing about which setting produced what, counted inventions out of 35:

                            judge 1   judge 2   judge 3
    unconstrained              2         7         6
    encoder_no_repeat = 6      8        20        15
    encoder_no_repeat = 4     20        26        23

All three found the gradient monotone. The fabrications were exactly what the blunt ban predicts:
a NDA company number rendered 00420028 for 01420028; an invented address
"correspondence team@verizon.co.uk" in place of a real Albany PO Box; a free-cash-flow figure of
$1.4B where the document says $0.7B; "$17.41 and $6.47" restated as the total when the document says
the total INCLUDES them. Denied the real digits, the model supplies plausible ones.

WHAT THIS DOES INSTEAD. The ban is applied to prose only. Any token that could carry a fact --
digits, currency, percentages, a capitalised word, a month -- is exempt, so the model is free to lift
"$236.60", "AT&T" and "16 April 2025" verbatim while still being pushed off the source's sentence
shapes. The constraint falls where paraphrase is wanted and lifts where accuracy is required.

This narrows the failure mode; it does not remove it. A model can still misattribute a figure it
copied correctly -- saying a total IS an amount the document says it INCLUDES -- and no
token-level rule can see that. pipeline/verify.py still runs, and a flag remains a prompt to read.
"""
import os
import re

import torch
from transformers.generation.logits_process import EncoderNoRepeatNGramLogitsProcessor

# A token worth protecting: it carries or begins a fact rather than joining a sentence.
FACTUAL = re.compile(r"[\d$£€%]|^\s*[A-Z]{2,}|^\s*[A-Z][a-z]+$")
MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")


def protected_token_ids(tok, cache={}):
    """Token ids that must never be banned. Computed once per tokenizer."""
    key = id(tok)
    if key in cache:
        return cache[key]
    keep = set()
    vocab = tok.get_vocab()
    for piece, tid in vocab.items():
        s = piece.replace("Ġ", " ").replace("##", "")
        t = s.strip()
        if not t:
            continue
        if FACTUAL.search(s) or t.lower()[:3] in MONTHS:
            keep.add(tid)
    cache[key] = keep
    return keep


class ProseOnlyNoRepeat(EncoderNoRepeatNGramLogitsProcessor):
    """encoder_no_repeat, except it never blocks a token that could carry a fact.

    The parent bans every continuation that would reproduce an input n-gram. This lifts the ban for
    the protected set, so a figure or a name can still be copied exactly while the surrounding prose
    is pushed away from the source's wording.
    """

    def __init__(self, ngram_size, encoder_input_ids, protected):
        super().__init__(ngram_size, encoder_input_ids)
        self.protected = protected

    def __call__(self, input_ids, scores):
        out = super().__call__(input_ids, scores)
        # wherever the parent banned a protected token, restore its original score
        banned = torch.isinf(out) & ~torch.isinf(scores)
        if banned.any():
            idx = torch.tensor(sorted(self.protected), device=scores.device, dtype=torch.long)
            idx = idx[idx < scores.shape[-1]]
            mask = torch.zeros(scores.shape[-1], dtype=torch.bool, device=scores.device)
            mask[idx] = True
            restore = banned & mask.unsqueeze(0)
            out = torch.where(restore, scores, out)
        return out


def processors(tok, encoder_input_ids, ngram=6):
    """The logits-processor list for prose-only abstraction, or empty to disable."""
    if not ngram:
        return []
    return [ProseOnlyNoRepeat(ngram, encoder_input_ids, protected_token_ids(tok))]
