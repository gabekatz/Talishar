python -m scripts.train --base-url http://localhost:8080 --p1-deck Ira --p2-deck Ira --total-steps 1000000

# Deck Downloader (scripts/download_deck.py)

Interactive CLI to browse and download competitive decks from
[fabrary.net](https://fabrary.net/most-played-decks) for use in AI training.

## Prerequisites

Playwright must be installed (included as a dev dependency):

```bash
uv sync --dev
uv run playwright install chromium
```

## Usage

```bash
# List all most-played decks (with hero names)
uv run python -m scripts.download_deck

# Search by hero name (case-insensitive substring match)
uv run python -m scripts.download_deck --hero "Dorinthea"
uv run python -m scripts.download_deck --hero "Kayo"
uv run python -m scripts.download_deck --hero "Fai"

# Scan more pages (default 6, ~63 decks/page)
uv run python -m scripts.download_deck --hero "Bravo" --pages 10
```

## Interactive flow

For each matching deck the tool prompts:

- **[i]nspect** — Print all cards in the deck (hero, equipment, main deck,
  sideboard) with Talishar card IDs and display names
- **[d]ownload** — Save the deck to `Assets/<Name>.txt` in the format expected
  by `CreateTrainingGame.php`, ready for `--p1-deck` / `--p2-deck`
- **[s]kip** — Move to the next matching deck
- **[q]uit** — Exit

## How it works

1. **Scraping**: Uses Playwright (headless Chromium) to load fabrary.net's
   JS-rendered pages. Hero names are extracted from hero image URLs in each
   deck listing entry.

2. **Card resolution**: Fabrary uses set-specific card codes (e.g. `WTR215`,
   `PEN319`). These are mapped to Talishar card IDs (e.g. `sink_below_red`,
   `command_and_conquer_red`) using `GeneratedCardDictionaries.php`:
   - **Direct lookup**: `GeneratedSetIDtoCardID` (4,600+ set-code mappings)
   - **Name fallback**: For newer sets not yet in the dictionary, falls back to
     `GeneratedCardName` reverse lookup with pitch-color heuristics

3. **Deck format**: Output matches the existing `Assets/*.txt` format:
   - Line 1: hero + equipment (space-separated)
   - Line 2: main deck cards (space-separated)
   - Lines 3+: sideboard cards (one per line)

## Limitations

- **Newer card sets** (e.g. SKA, SDO, SFA) may not be fully mapped in
  `GeneratedCardDictionaries.php`. The tool warns about unmapped cards.
- **Pitch color** for cross-set reprints uses a heuristic that can be wrong.
  The card will be valid but may be the wrong pitch variant.
- **fabrary.net availability**: The tool depends on fabrary.net's page
  structure. If they redesign the site, the scraper selectors will need
  updating.