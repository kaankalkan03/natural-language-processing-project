# Data license and attribution

The MIT license in `LICENSE` covers the code only. This file covers the data.

## Source

Transcripts of the FOMC Chair's press conferences, published by the Board of
Governors of the Federal Reserve System, April 2011 to September 2026
(95 PDF files, downloaded on 22 October 2026). Each file's URL, size,
SHA-256 and retrieval time are listed in `data/raw/manifest.csv`.

## Terms

The Board's disclaimer page (https://www.federalreserve.gov/disclaimer.htm,
checked on 23 October 2026) states that information on the Board's website is
in the public domain and may be copied and distributed without permission. The
exception is material identified as coming from a non-Board source, such as
photos or graphics with copyright or trademark restrictions; the transcripts
are text published by the Board. The page asks users to cite the Board as the
source of the information.

This applies to:

- `data/raw/pdf/` (the transcripts as published),
- `data/clean/` and `data/processed/` (text derived from them by our cleaning
  and labelling scripts).

## Citation

Board of Governors of the Federal Reserve System. *FOMC Press Conference
Transcripts*, 2011–2026.
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Our derived files (cleaned turns, labels, splits) can be shared on the same
terms; please cite the Board as the source of the text and this repository for
the cleaning and labelling.
