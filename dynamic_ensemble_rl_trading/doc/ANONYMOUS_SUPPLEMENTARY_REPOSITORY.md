# Anonymous Supplementary Repository — FAQ for Reviewers

**Manuscript ID:** ESWA-D-26-08980

During double-blind peer review, supplementary code is **not** distributed via an author-identified public GitHub account. Instead, the editorial office (or the authors' cover letter) supplies an **anonymous mirror link** or an **offline ZIP bundle**.

---

## Why does the anonymous URL contain `-6EED` (or similar)?

If you received a link such as:

`https://anonymous.4open.science/r/SOME-NAME-6EED/`

the suffix **`-6EED` is not a password, author ID, or hidden tracking code**. It is a **random unique suffix automatically assigned** by [Anonymous GitHub](https://anonymous.4open.science/) when the mirror is created, so that millions of anonymized repositories never collide.

| Part of URL | Meaning |
|-------------|---------|
| `anonymous.4open.science` | Third-party anonymization service (not GitHub) |
| `r/SOME-NAME-6EED` | System-assigned anonymous repository ID |
| `-6EED` | Random hex fragment — **ignore as metadata**; not credentials |

Reviewers should treat the full URL as an opaque download link, similar to a DOI or supplementary-file handle.

---

## Where to start inside the mirror

Always open the **audit project folder**, not any legacy stub at the repository root:

```
dynamic_ensemble_rl_trading/
├── README.md                 ← Honesty Statement (start here)
├── doc/REVIEWER_INDEX.md     ← navigation index
├── reproduce.py              ← one-command audit reproduction
└── results/audit/            ← pre-computed headline JSON/MD
```

**Entry URL pattern (example structure only):**

`…/dynamic_ensemble_rl_trading/doc/REVIEWER_INDEX.md`

---

## What is redacted in this release?

- Author names and signatures in letters bundled for blind review
- Author-linked GitHub usernames and clone URLs
- Internal advisor meeting notes (excluded from upload)

Author identity is known to the Editor via Editorial Manager; it must **not** be inferred from this supplementary bundle during active review.

---

## Offline alternative

Authors may also attach **`ESWA-D-26-08980_reviewer_code.zip`**. Unzip, open `REVIEWER_QUICKSTART.txt`, then run:

```bash
python reproduce.py --only ci
```

---

## After acceptance

A permanent, author-attributed public repository URL may replace the anonymous mirror in the final published article. Until then, use only the link or ZIP supplied for review.
