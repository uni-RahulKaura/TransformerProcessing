export const meta = {
  name: 'grade-outlines-blind',
  description: 'Three specialised blind judges per outline, adversarial verification, then one verdict',
  phases: [
    { title: 'Judge',     detail: 'missing sections, spurious sections, hierarchy -- each against the raw Markdown' },
    { title: 'Verify',    detail: 'three skeptics per high-severity finding, majority rules' },
    { title: 'Synthesis', detail: 'one verdict on outline quality' },
  ],
}

const ROOT = '/Users/rahulkaura/Documents/gap-format-prototype'

const BAN = `
HARD RULES -- breaking any of these invalidates your work:
* You may read ONLY the raw Markdown named in your packet and the outline file named in your packet.
* You must NOT read, list or grep any of: ${ROOT}/indexer*, ${ROOT}/gpu-bundle*/pipeline,
  ${ROOT}/gpu-bundle*/*.py, ${ROOT}/corpus, ${ROOT}/judge/*.js, ${ROOT}/judge/packets*,
  any *.json under ${ROOT}/judge/outlines, ${ROOT}/BBC-DEMO, ${ROOT}/RUN-GPU*, any *ground_truth*,
  any *.truth.txt, or any other judge's output. Those contain the code that produced the outline and
  other judges' opinions; either one biases you.
* The raw Markdown is the sole authority. Where it and the outline disagree, the Markdown is right.
* Quote exactly. Every finding must carry the literal text from the raw Markdown. A finding you
  cannot quote is not a finding -- drop it.
* An empty findings list is a valid and useful answer. Prefer it to a speculative one.

HOW TO READ THE RAW MARKDOWN
It is Landing AI output. Some conventions you need so you do not mistake formatting for content:
* <a id='...'></a> marks where a visual block began on the page. It is not a heading.
* <::  ...  ::> is a vision model's description of a chart, diagram or signature block. Not a heading.
* <table>...</table> is a real table. Cell text is NOT a heading, however capitalised it looks.
* A line in **bold** may or may not be a heading -- judge it on whether it introduces the text under it.
`

const FINDINGS = {
  type: 'object', additionalProperties: false,
  properties: {
    outline_entries_checked: { type: 'integer' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          claim: { type: 'string' },
          quote_from_raw_markdown: { type: 'string' },
          outline_entry: { type: 'string', description: 'the outline line at issue, or "(absent)"' },
          why_wrong: { type: 'string' },
        },
        required: ['severity', 'claim', 'quote_from_raw_markdown', 'outline_entry', 'why_wrong'],
      },
    },
  },
  required: ['outline_entries_checked', 'findings'],
}

const VERDICT = {
  type: 'object', additionalProperties: false,
  properties: {
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
    confidence: { type: 'string', enum: ['low', 'medium', 'high'] },
  },
  required: ['refuted', 'reason', 'confidence'],
}

const REMITS = [
  { key: 'missing', brief: `You look for REAL SECTIONS THE OUTLINE DOES NOT HAVE.

Read the raw Markdown and identify every place a new section genuinely begins -- a numbered clause, a
titled heading, a bolded lead-in that introduces the text beneath it, an attachment or exhibit that
starts. Then check each against the outline. Report the ones that are absent.

Do not report as missing: a table's caption, a figure's caption, a page header or footer, a DocuSign
envelope line, a page number, or a line of body text that merely happens to be capitalised. Those are
not sections and the outline is right to omit them.` },

  { key: 'spurious', brief: `You look for OUTLINE ENTRIES THAT ARE NOT REAL SECTIONS.

For each entry in the outline, find the corresponding text in the raw Markdown and decide whether it
genuinely begins a section. Report entries that do not: a table caption or a cell's contents promoted
to a heading, a figure caption, a page header or footer, a fragment of a sentence, a party's address
line, a signature-block label, or a heading that is really the tail of the previous one split in two.

Quote the raw Markdown around the entry so the claim can be checked.` },

  { key: 'hierarchy', brief: `You check NESTING ONLY.

The outline indents children under parents. Using the raw Markdown's own numbering and layout, decide
whether each entry sits under the right parent. Report an entry nested under the wrong parent, an
entry that should be nested but sits at the top level, an entry pushed a level too deep, and any place
where the document's numbering (2.1 under 2, 3.7.1 under 3.7) is contradicted by the indentation.

Do not report the absence or presence of a section -- other judges hold that remit. Nesting only.` },
]

const packets = (args && args.packets) || []
log(`grading ${packets.length} outlines with ${REMITS.length} specialised judges each`)

const perDoc = await pipeline(
  packets,
  (pk, orig, i) => parallel(REMITS.map(r => () =>
    agent(`You are one of several independent judges of a document outline. You have one remit.

YOUR REMIT
${r.brief}

${BAN}

WHAT TO DO
1. Read the packet ${pk}. It names the raw Markdown and the outline file, and says how many entries
   the outline has.
2. Read BOTH files in full.
3. Apply your remit. Set outline_entries_checked to the number you actually examined.

Return the structured object. Your prose output is discarded -- only the structured result is used.`,
      { label: `${r.key}:${i + 1}`, phase: 'Judge', schema: FINDINGS })
      .then(v => ({ remit: r.key, packet: orig, result: v })))),

  (judged, orig, i) => {
    const found = (judged || []).filter(Boolean)
    const claims = []
    for (const j of found) {
      for (const f of ((j.result && j.result.findings) || [])) {
        if (f.severity === 'critical' || f.severity === 'high') claims.push({ ...f, remit: j.remit })
      }
    }
    if (!claims.length) return { packet: orig, judged: found, confirmed: [], claims: 0, verified: 0 }
    const CAP = 12
    const picked = claims.slice(0, CAP)
    if (claims.length > CAP) {
      log(`${orig.split('/').pop()}: ${claims.length} high/critical outline claims, verifying ${CAP} -- ${claims.length - CAP} NOT verified`)
    }
    return parallel(picked.map(c => () =>
      parallel([0, 1, 2].map(k => () =>
        agent(`You are a SKEPTIC. Another judge claims a document's outline is wrong. REFUTE the claim if it
can be refuted. Default to refuted=true unless the claim clearly holds.

The claim is REFUTED if any of these is true:
* the quoted text is a table caption, figure caption, page header, footer, page number, DocuSign
  envelope line, address line or signature label -- none of which is a section
* the quoted text is body prose that merely looks like a heading because it is short or capitalised
* the outline does contain the section, under a slightly different title or at a different indent
* the disagreement is about wording of a title rather than whether the section exists
* the nesting the judge objects to is in fact what the document's own numbering implies

The claim STANDS only if a reader using this outline to navigate the document would be sent to the
wrong place or would not find a section that is really there.

${BAN}

THE CLAIM
  remit           : ${c.remit}
  claim           : ${c.claim}
  quoted Markdown : ${c.quote_from_raw_markdown}
  outline entry   : ${c.outline_entry}
  reasoning       : ${c.why_wrong}

Read the two files named in ${orig} and decide. Angle ${k + 1} of 3:
${['check whether the quoted text is structural or is furniture such as a caption, header or cell',
   'check whether the outline contains the section elsewhere, under another title or indent',
   'check whether a reader navigating by this outline would actually be misled'][k]}.

Return the structured verdict.`, { label: `refute${k + 1}:${i + 1}`, phase: 'Verify', schema: VERDICT })))
        .then(vs => {
          const v = vs.filter(Boolean)
          return { ...c, stands: v.filter(x => !x.refuted).length >= 2,
                   votes: v.map(x => ({ refuted: x.refuted, why: (x.reason || '').slice(0, 200) })) }
        })
    )).then(rs => ({
      packet: orig, judged: found,
      confirmed: rs.filter(Boolean).filter(r => r.stands),
      claims: claims.length, verified: picked.length,
    }))
  }
)

phase('Synthesis')
const docs = perDoc.filter(Boolean)
const byRemit = {}
for (const d of docs) for (const j of (d.judged || [])) {
  byRemit[j.remit] = byRemit[j.remit] || { checked: 0, raised: 0 }
  byRemit[j.remit].checked += (j.result && j.result.outline_entries_checked) || 0
  byRemit[j.remit].raised += ((j.result && j.result.findings) || []).length
}
const totals = {
  outlines: docs.length,
  by_remit: byRemit,
  high_or_critical_claims: docs.reduce((a, d) => a + (d.claims || 0), 0),
  claims_verified: docs.reduce((a, d) => a + (d.verified || 0), 0),
  confirmed_after_three_skeptics: docs.reduce((a, d) => a + (d.confirmed || []).length, 0),
}
const confirmed = docs.flatMap(d => (d.confirmed || []).map(c => ({
  document: d.packet.split('/').pop().replace('.outline-packet.json', ''),
  remit: c.remit, severity: c.severity, claim: c.claim,
  raw: (c.quote_from_raw_markdown || '').slice(0, 260),
  entry: (c.outline_entry || '').slice(0, 160),
})))
log(`outline defects confirmed: ${totals.confirmed_after_three_skeptics} of ${totals.high_or_critical_claims} claims`)

const report = await agent(`Write the verdict on a document outliner, for the engineer who owns it.

You are given every outline defect that survived three independent skeptics, plus counts.

MEASUREMENTS
${JSON.stringify(totals, null, 1)}

CONFIRMED DEFECTS
${JSON.stringify(confirmed, null, 1)}

Write in plain prose, no marketing language:
1. Are these outlines trustworthy for navigating the documents? State the number that decides it.
2. The defect classes that survived, worst first, each with one quoted example. Group repeats and give
   counts -- if one class appears in six documents, say so once.
3. Whether the defects cluster by document type, and if so which type is worst and why that follows
   from its shape.
4. What to fix, in priority order, each tied to a class above. Name the change, not the goal.
5. What this did NOT establish. Be specific about coverage: how many claims went unverified because of
   the per-document cap, and what a judge reading only the Markdown could not see.

Do not congratulate anyone. If something is broken, say so.`,
  { label: 'outline-verdict', phase: 'Synthesis' })

return { totals, confirmed, report }
