"""
scripts/download_deck.py — CLI to browse and download decks from fabrary.net
for use in Talishar AI training.

Usage
-----
# Interactive: browse by hero
python -m talishar_ai.scripts.download_deck --hero "Ira"
python -m talishar_ai.scripts.download_deck              # lists all decks

# Bulk: auto-download best deck per unique hero
python -m talishar_ai.scripts.download_deck --bulk
python -m talishar_ai.scripts.download_deck --bulk --format cc
python -m talishar_ai.scripts.download_deck --bulk --format blitz
python -m talishar_ai.scripts.download_deck --bulk --format silver-age

# Interactive with format filter
python -m talishar_ai.scripts.download_deck --hero "Bravo" --format cc

The tool scrapes https://fabrary.net/most-played-decks, lets you browse by
hero, inspect deck contents, and download decks to Assets/ in the format
expected by CreateTrainingGame.php.

In --bulk mode, it automatically downloads the highest win% deck for each
unique hero found in the listing, saving each to Assets/<HeroName>.txt.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Set-code → Talishar card-ID mapping (parsed from GeneratedCardDictionaries)
# ---------------------------------------------------------------------------

_SET_ID_MAP: dict[str, str] | None = None
_NAME_TO_IDS: dict[str, list[str]] | None = None


def _parse_php_match(func_name: str) -> dict[str, str]:
    """Parse a PHP match() block from GeneratedCardDictionaries.php."""
    php_path = Path(__file__).resolve().parents[2] / "GeneratedCode" / "GeneratedCardDictionaries.php"
    if not php_path.exists():
        print(f"[ERROR] Cannot find {php_path}")
        sys.exit(1)

    mapping: dict[str, str] = {}
    in_func = False
    with open(php_path, "r") as f:
        for line in f:
            if f"function {func_name}" in line:
                in_func = True
                continue
            if in_func:
                if line.strip().startswith("};") or line.strip().startswith("default =>"):
                    break
                m = re.match(r'\s*"([^"]+)"\s*=>\s*"([^"]+)"', line)
                if m:
                    mapping[m.group(1)] = m.group(2)
    return mapping


def _load_set_id_map() -> dict[str, str]:
    """Parse GeneratedSetIDtoCardID from the PHP source into a Python dict."""
    global _SET_ID_MAP
    if _SET_ID_MAP is not None:
        return _SET_ID_MAP
    _SET_ID_MAP = _parse_php_match("GeneratedSetIDtoCardID")
    print(f"[deck] Loaded {len(_SET_ID_MAP):,} set-code → card-ID mappings")
    return _SET_ID_MAP


def _load_name_to_ids() -> dict[str, list[str]]:
    """Build a display-name → [card_id, ...] reverse lookup."""
    global _NAME_TO_IDS
    if _NAME_TO_IDS is not None:
        return _NAME_TO_IDS

    name_map = _parse_php_match("GeneratedCardName")
    # name_map: card_id → display_name.  Invert to name → [card_ids].
    result: dict[str, list[str]] = {}
    for card_id, display_name in name_map.items():
        key = display_name.lower()
        result.setdefault(key, []).append(card_id)
    _NAME_TO_IDS = result
    print(f"[deck] Loaded {len(result):,} card-name → ID reverse mappings")
    return result


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def _fetch_deck_list(max_pages: int = 6, format_filter: str | None = None) -> list[dict]:
    """
    Fetch the most-played decks listing from fabrary.net.

    Parameters
    ----------
    max_pages : int
        Number of pagination pages to scrape.
    format_filter : str | None
        Game format to filter by (e.g. "cc", "blitz", "living legend").
        The scraper will try to click the matching format tab/button on
        the page.  If no UI filter is found, decks are filtered client-side
        by checking format text in each deck entry.

    Returns a list of dicts with keys: name, hero, href, url, win_rate, format.
    """
    from playwright.sync_api import sync_playwright

    decks: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(
            "https://fabrary.net/most-played-decks",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        page.wait_for_selector('a[href*="/decks/"]', timeout=15000)

        # Try to click a format filter on the page if requested
        format_clicked = False
        if format_filter:
            format_clicked = _try_click_format_filter(page, format_filter)

        for pg in range(1, max_pages + 1):
            print(f"[deck] Fetching page {pg}...", end=" ", flush=True)

            page_decks = _extract_deck_links(page)
            print(f"{len(page_decks)} decks")
            decks.extend(page_decks)

            if not page_decks:
                break

            # Navigate to next page via the "Next" button (JS-driven pagination)
            if pg < max_pages:
                next_btn = page.query_selector('button:has-text("Next")')
                if not next_btn or next_btn.is_disabled():
                    print("[deck] No more pages.")
                    break
                next_btn.click()
                # Wait for the page content to refresh
                page.wait_for_timeout(2000)
                page.wait_for_selector('a[href*="/decks/"]', timeout=15000)

        browser.close()

    # Client-side format filtering if we couldn't click a UI filter
    if format_filter and not format_clicked:
        norm = format_filter.lower().replace("-", " ").replace("_", " ")
        before = len(decks)
        decks = [d for d in decks if _format_matches(d.get("format", ""), norm)]
        print(f"[deck] Format filter '{format_filter}': {before} → {len(decks)} decks")

    return decks


def _try_click_format_filter(page, format_filter: str) -> bool:
    """Try to find and click a format filter button/tab/dropdown on the page."""
    norm = format_filter.lower().replace("-", " ").replace("_", " ")

    # Build list of exact labels to look for (all common capitalizations)
    labels = {norm}
    aliases = {
        "cc": ["CC", "Classic Constructed"],
        "classic constructed": ["CC", "Classic Constructed"],
        "blitz": ["Blitz"],
        "living legend": ["Living Legend", "LL"],
        "ll": ["Living Legend", "LL"],
        "commoner": ["Commoner"],
        "silver age": ["Silver Age"],
    }
    for key, vals in aliases.items():
        if norm == key:
            labels.update(vals)
            labels.add(key)

    # Strategy 1: Find buttons/tabs/links whose trimmed text exactly matches
    for tag in ["button", "a", '[role="tab"]']:
        try:
            elements = page.query_selector_all(tag)
            for el in elements:
                try:
                    text = (el.inner_text() or "").strip()
                except Exception:
                    continue
                if text in labels or text.lower() in {l.lower() for l in labels}:
                    if el.is_visible():
                        el.click()
                        page.wait_for_timeout(2000)
                        print(f"[deck] Clicked format filter: '{text}'")
                        return True
        except Exception:
            pass

    # Strategy 2: Look for a select/dropdown containing format options
    try:
        selects = page.query_selector_all("select")
        for sel in selects:
            options = sel.query_selector_all("option")
            for opt in options:
                text = (opt.inner_text() or "").strip()
                if text in labels or text.lower() in {l.lower() for l in labels}:
                    sel.select_option(label=text)
                    page.wait_for_timeout(2000)
                    print(f"[deck] Selected format from dropdown: '{text}'")
                    return True
    except Exception:
        pass

    print(f"[deck] No format filter UI found for '{format_filter}' — will filter client-side")
    return False


def _format_matches(deck_format: str, query: str) -> bool:
    """Check if a deck's format matches the query (fuzzy)."""
    if not deck_format:
        return True  # no format info available — include by default
    df = deck_format.lower().replace("-", " ").replace("_", " ")
    # Handle common abbreviations
    aliases = {
        "cc": ["classic constructed", "cc"],
        "blitz": ["blitz"],
        "living legend": ["living legend", "ll"],
        "commoner": ["commoner"],
        "silver age": ["silver age"],
    }
    for canonical, names in aliases.items():
        if query in names or query == canonical:
            return any(n in df for n in names) or canonical in df
    return query in df


def _extract_deck_links(page) -> list[dict]:
    """Extract real deck links from the current page, filtering out action buttons."""
    import re as _re

    links = page.query_selector_all('a[href*="/decks/"]')
    page_decks = []
    seen_hrefs: set[str] = set()

    for link in links:
        href = link.get_attribute("href") or ""

        # Filter out non-deck links:
        # - "Play on Talishar" → href starts with http (external link)
        # - "Compare with other decks" → href contains /decks/compare
        # - Must match /decks/{ULID} pattern (26-char alphanumeric ID)
        if not href or href.startswith("http") or "compare" in href:
            continue
        if not _re.match(r"^/decks/[A-Z0-9]{20,}$", href):
            continue
        if href in seen_hrefs:
            continue
        seen_hrefs.add(href)

        name = link.inner_text().split("\n")[0].strip()
        if not name:
            continue

        # Extract hero and win rate from the parent container.
        # Hero: from hero image URL (content.fabrary.net/heroes/<slug>.webp)
        # Win rate: from text like "3,190 / 6,152 (52%)"
        hero = ""
        win_rate = 0.0
        try:
            info = link.evaluate('''el => {
                let p = el;
                for (let i = 0; i < 8; i++) {
                    p = p.parentElement;
                    if (!p) break;
                }
                let hero = "";
                let winRate = 0;
                let format = "";
                if (p) {
                    const imgs = p.querySelectorAll("img");
                    for (const img of imgs) {
                        const src = img.src || "";
                        if (src.includes("/heroes/")) {
                            hero = src.split("/heroes/")[1].replace(".webp", "");
                            break;
                        }
                    }
                    const text = p.innerText || "";
                    const m = text.match(/(\\d+)%/);
                    if (m) winRate = parseInt(m[1]);
                    // Try to find format info (CC, Blitz, etc.)
                    const formatPatterns = ["Classic Constructed", "CC", "Blitz", "Living Legend", "Commoner", "Silver Age"];
                    for (const fp of formatPatterns) {
                        if (text.includes(fp)) {
                            format = fp;
                            break;
                        }
                    }
                    // Also check for format badges/spans
                    const badges = p.querySelectorAll("span, badge, .badge, .tag, .format, .label");
                    for (const b of badges) {
                        const bt = (b.innerText || "").trim();
                        for (const fp of formatPatterns) {
                            if (bt === fp || bt.toLowerCase() === fp.toLowerCase()) {
                                format = fp;
                                break;
                            }
                        }
                        if (format) break;
                    }
                }
                return {hero: hero, winRate: winRate, format: format};
            }''')
            hero = info["hero"].replace("-", " ").title() if info.get("hero") else ""
            win_rate = float(info.get("winRate", 0))
            fmt = info.get("format", "")
        except Exception:
            pass

        page_decks.append({
            "name": name,
            "hero": hero,
            "href": href,
            "url": f"https://fabrary.net{href}",
            "win_rate": win_rate,
            "format": fmt,
        })

    return page_decks


def _fetch_deck_detail(url: str) -> dict:
    """
    Fetch a single deck page and extract hero, equipment, deck, and inventory.

    Returns a dict with keys:
      hero_arena: list of {code, name}  (hero + equipment)
      deck:       list of {code, name}  (main deck)
      inventory:  list of {code, name}  (sideboard / inventory)
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Wait for card images to load
        page.wait_for_selector('img[src*="content.fabrary.net/cards/"]', timeout=15000)
        # Give a moment for all sections to render
        page.wait_for_timeout(1000)

        sections: dict[str, list[dict]] = {}
        current_key: str | None = None

        # Walk the page: find section headers then collect card images
        elements = page.query_selector_all("*")
        for el in elements:
            try:
                tag = el.evaluate("e => e.tagName")
            except Exception:
                continue

            if tag == "IMG":
                src = el.get_attribute("src") or ""
                alt = el.get_attribute("alt") or ""
                if "content.fabrary.net/cards/" in src and current_key:
                    code = src.split("/cards/")[1].replace(".webp", "")
                    sections[current_key].append({"code": code, "name": alt})
            else:
                text = ""
                try:
                    text = el.inner_text().strip()
                except Exception:
                    continue
                # Match section headers like "Hero + arena (5)", "Deck (44)", "Inventory (31)"
                if re.match(r"^(Hero \+ arena|Deck|Inventory|Sideboard|Equipment|Weapons?)\s*\(\d+\)$", text, re.I):
                    key = text.split("(")[0].strip().lower().replace(" + ", "_")
                    current_key = key
                    if key not in sections:
                        sections[key] = []
                # Stop collecting when we hit "Maybe"
                elif re.match(r"^Maybe\s*\(\d+\)$", text, re.I):
                    current_key = None

        browser.close()

    return {
        "hero_arena": sections.get("hero_arena", []),
        "deck": sections.get("deck", []),
        "inventory": sections.get("inventory", sections.get("sideboard", [])),
    }


# ---------------------------------------------------------------------------
# Conversion to Talishar format
# ---------------------------------------------------------------------------

def _resolve_card_id(code: str, name: str, set_map: dict[str, str]) -> str | None:
    """Convert a fabrary set code to a Talishar card ID.

    Strategy:
    1. Direct set-code lookup (fastest, most precise)
    2. Zero-padding variants of the set code
    3. Name-based fallback: match display name, pick the best variant
       (cards with color suffixes like _red/_yellow/_blue are disambiguated
       by their position in the set — lower numbers are typically red).
    """
    # 1. Direct lookup
    card_id = set_map.get(code)
    if card_id:
        return card_id

    # 2. Zero-padding variants
    m_code = re.match(r"([A-Z]+)0*(\d+)", code)
    if m_code:
        prefix, num = m_code.group(1), m_code.group(2)
        for n_digits in range(len(num), len(num) + 3):
            alt = f"{prefix}{num.zfill(n_digits)}"
            if alt in set_map:
                return set_map[alt]

    # 3. Name-based fallback
    if not name:
        return None
    name_map = _load_name_to_ids()
    candidates = name_map.get(name.lower(), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Multiple pitch variants exist.  Use set-code number to guess pitch:
        # In most FaB sets, red < yellow < blue within a card's numbering.
        # We try to match based on the code's numeric offset from the base.
        if m_code:
            num_val = int(m_code.group(2))
            # Sort candidates by color suffix: red=0, yellow=1, blue=2, none=0
            def _color_order(cid: str) -> int:
                if cid.endswith("_red"): return 0
                if cid.endswith("_yellow"): return 1
                if cid.endswith("_blue"): return 2
                return -1

            colored = [c for c in candidates if _color_order(c) >= 0]
            uncolored = [c for c in candidates if _color_order(c) < 0]

            if colored:
                colored.sort(key=_color_order)
                # Check if there are other set codes for this card's variants
                # to determine offset
                base_codes = []
                for cand in colored:
                    for sc, cid in set_map.items():
                        if cid == cand:
                            base_codes.append((sc, cand))
                            break

                if base_codes:
                    # Sort by color order so red=0, yellow=1, blue=2
                    base_codes.sort(key=lambda x: _color_order(x[1]))

                    # Check same-prefix exact match first
                    for sc, cid in base_codes:
                        bc_match = re.match(r"([A-Z]+)(\d+)", sc)
                        if bc_match and bc_match.group(1) == (m_code.group(1) if m_code else ""):
                            if int(bc_match.group(2)) == num_val:
                                return cid

                    # Cross-prefix: use relative offset from lowest known
                    # set code to determine pitch.  In FaB, within a card's
                    # variants the numbering goes red, yellow, blue (+0,+1,+2).
                    # Find the minimum set-code number for same prefix among
                    # ALL cards with this name, to compute offset.
                    same_prefix = [
                        (sc, cid) for sc, cid in base_codes
                        if re.match(r"[A-Z]+", sc).group() == m_code.group(1)
                    ] if m_code else []

                    if same_prefix:
                        same_prefix.sort(key=lambda x: int(re.search(r"\d+", x[0]).group()))
                        base_num = int(re.search(r"\d+", same_prefix[0][0]).group())
                        offset = num_val - base_num
                        if 0 <= offset < len(colored):
                            return colored[offset]
                    else:
                        # No same-prefix codes known.  Use the number's offset
                        # from the lowest code number for this card in ANY set.
                        all_nums = []
                        for sc, cid in base_codes:
                            nm = re.search(r"\d+", sc)
                            if nm:
                                all_nums.append(int(nm.group()))
                        if all_nums:
                            # Guess: last digit or mod-3 of difference
                            min_num = min(all_nums)
                            # Cards are typically sequential: num, num+1, num+2
                            # for red, yellow, blue.  Use modulo of our code.
                            offset = num_val % 3  # rough heuristic
                            # Better: if our code ends in 6/7/8 pattern etc,
                            # use offset from group of 3
                            offset = (num_val - min_num) % len(colored)
                            if 0 <= offset < len(colored):
                                return colored[offset]

                # Fallback: just return the first colored variant (red)
                return colored[0]

            if uncolored:
                return uncolored[0]

        # Last resort: return first candidate
        return candidates[0]

    return None


def _build_deck_file(detail: dict, set_map: dict[str, str]) -> tuple[str, list[str]]:
    """
    Convert scraped deck detail to Talishar Assets/ format.

    Returns (file_content, list_of_warnings).

    Talishar deck format:
      Line 1: hero weapon1 weapon2 head chest arms legs  (space-separated, 7 items)
      Line 2: main deck cards (space-separated)
      Line 3: (empty)
      Lines 4+: sideboard cards (one per line)
    """
    warnings: list[str] = []

    # --- Hero + arena ---
    hero_ids: list[str] = []
    for card in detail["hero_arena"]:
        cid = _resolve_card_id(card["code"], card["name"], set_map)
        if cid:
            hero_ids.append(cid)
        else:
            warnings.append(f"Unknown hero/equipment: {card['code']} ({card['name']})")

    # Pad to 7 items (hero + 6 equipment slots) if needed
    # The order is: hero, weapon, weapon, head, chest, arms, legs
    # We can't reliably determine slot order from fabrary, but the engine
    # reads them positionally. Usually hero is first, then equipment.
    if len(hero_ids) < 7:
        warnings.append(f"Only {len(hero_ids)} hero/equipment items resolved (7 slots expected)")
        while len(hero_ids) < 7:
            hero_ids.append("")

    # --- Main deck ---
    deck_ids: list[str] = []
    for card in detail["deck"]:
        cid = _resolve_card_id(card["code"], card["name"], set_map)
        if cid:
            deck_ids.append(cid)
        else:
            warnings.append(f"Unknown deck card: {card['code']} ({card['name']})")

    # --- Inventory / sideboard ---
    inv_ids: list[str] = []
    for card in detail["inventory"]:
        cid = _resolve_card_id(card["code"], card["name"], set_map)
        if cid:
            inv_ids.append(cid)
        else:
            warnings.append(f"Unknown inventory card: {card['code']} ({card['name']})")

    # Build file content
    lines = [
        " ".join(hero_ids).strip(),
        " ".join(deck_ids),
        "",  # blank line
    ]
    for inv in inv_ids:
        lines.append(inv)

    # Ensure trailing newlines match existing format
    content = "\n".join(lines) + "\n"
    return content, warnings


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_deck_detail(detail: dict, set_map: dict[str, str]) -> None:
    """Pretty-print a deck's contents."""
    print()
    print(f"  Hero + Arena ({len(detail['hero_arena'])} cards):")
    for card in detail["hero_arena"]:
        cid = _resolve_card_id(card["code"], card["name"], set_map) or "???"
        print(f"    {cid:40s}  ({card['name']})")

    print(f"\n  Main Deck ({len(detail['deck'])} cards):")
    # Group by card ID for readability
    from collections import Counter
    deck_counts: Counter[str] = Counter()
    deck_names: dict[str, str] = {}
    for card in detail["deck"]:
        cid = _resolve_card_id(card["code"], card["name"], set_map) or f"???:{card['code']}"
        deck_counts[cid] += 1
        deck_names[cid] = card["name"]
    for cid, count in sorted(deck_counts.items()):
        print(f"    {count}x {cid:40s}  ({deck_names[cid]})")

    if detail["inventory"]:
        print(f"\n  Inventory / Sideboard ({len(detail['inventory'])} cards):")
        inv_counts: Counter[str] = Counter()
        inv_names: dict[str, str] = {}
        for card in detail["inventory"]:
            cid = _resolve_card_id(card["code"], card["name"], set_map) or f"???:{card['code']}"
            inv_counts[cid] += 1
            inv_names[cid] = card["name"]
        for cid, count in sorted(inv_counts.items()):
            print(f"    {count}x {cid:40s}  ({inv_names[cid]})")
    print()


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def _generate_filename(detail: dict, set_map: dict[str, str]) -> str:
    """Generate a filename from the hero card in a deck detail."""
    hero_id = _resolve_card_id(
        detail["hero_arena"][0]["code"],
        detail["hero_arena"][0]["name"],
        set_map,
    )
    default_name = hero_id or detail["hero_arena"][0]["name"].replace(" ", "_")
    default_name = default_name.replace(",", "").replace("'", "")
    return "".join(w.capitalize() for w in default_name.split("_"))


def _bulk_download(
    decks: list[dict], set_map: dict[str, str], assets_dir: Path,
) -> None:
    """
    Download the highest win% deck for each unique hero.

    Groups decks by hero, picks the best one per hero, fetches detail,
    and saves to Assets/.
    """
    # Group by hero, pick best win% per hero
    best_by_hero: dict[str, dict] = {}
    for d in decks:
        hero = d.get("hero", "").strip()
        if not hero:
            continue
        if hero not in best_by_hero or d.get("win_rate", 0) > best_by_hero[hero].get("win_rate", 0):
            best_by_hero[hero] = d

    if not best_by_hero:
        print("[ERROR] No decks with hero info found.")
        return

    # Sort by hero name for readable output
    heroes = sorted(best_by_hero.keys())
    print(f"\n[deck] Found {len(heroes)} unique heroes. Downloading best deck for each...\n")

    downloaded = 0
    skipped = 0
    errors = 0

    for i, hero in enumerate(heroes, 1):
        deck = best_by_hero[hero]
        wr = deck.get("win_rate", 0)
        fmt = deck.get("format", "")
        fmt_tag = f" [{fmt}]" if fmt else ""
        print(f"  [{i}/{len(heroes)}] {hero} — {wr:.0f}%{fmt_tag} — {deck['name']}")

        try:
            detail = _fetch_deck_detail(deck["url"])
            if not detail["hero_arena"]:
                print(f"    SKIP: no hero/equipment found")
                skipped += 1
                continue

            content, warnings = _build_deck_file(detail, set_map)
            for w in warnings:
                print(f"    [WARN] {w}")

            filename = _generate_filename(detail, set_map)
            out_path = assets_dir / f"{filename}.txt"

            # Don't overwrite without noting it
            if out_path.exists():
                print(f"    Overwriting {out_path.name}")

            out_path.write_text(content)
            print(f"    Saved: Assets/{filename}.txt")
            downloaded += 1

        except Exception as exc:
            print(f"    ERROR: {exc}")
            errors += 1

    print(f"\n[deck] Bulk download complete: {downloaded} saved, {skipped} skipped, {errors} errors")
    print(f"[deck] Deck files in: {assets_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse and download decks from fabrary.net for Talishar AI training."
    )
    parser.add_argument(
        "--hero", default=None,
        help="Hero name to search for (case-insensitive substring match). "
             "If omitted, lists all available decks."
    )
    parser.add_argument(
        "--pages", type=int, default=6,
        help="Number of most-played-decks pages to fetch (default: 6)."
    )
    parser.add_argument(
        "--bulk", action="store_true",
        help="Bulk download: automatically download the highest win%% deck "
             "for each unique hero. No interactive prompts."
    )
    parser.add_argument(
        "--format", default=None, dest="game_format",
        help="Filter decks by game format (e.g. 'cc', 'blitz', 'living-legend', "
             "'silver-age', 'commoner'). Works with both --bulk and interactive mode."
    )
    args = parser.parse_args()

    set_map = _load_set_id_map()

    print("[deck] Fetching most-played decks from fabrary.net...")
    all_decks = _fetch_deck_list(max_pages=args.pages, format_filter=args.game_format)

    if not all_decks:
        print("[ERROR] No decks found.")
        return

    # --- Bulk mode ---
    if args.bulk:
        assets_dir = Path(__file__).resolve().parents[2] / "Assets"
        _bulk_download(all_decks, set_map, assets_dir)
        return

    # --- Interactive mode ---
    if args.hero is None:
        all_decks.sort(key=lambda d: d.get("win_rate", 0), reverse=True)
        print(f"\n  Found {len(all_decks)} decks (sorted by win rate). Use --hero to filter.\n")
        for i, d in enumerate(all_decks):
            hero_tag = f" [{d['hero']}]" if d.get("hero") else ""
            fmt_tag = f" ({d['format']})" if d.get("format") else ""
            wr = d.get("win_rate", 0)
            print(f"  {i+1:3d}. {wr:2.0f}% | {d['name']}{hero_tag}{fmt_tag}")
        return

    hero_query = args.hero.lower()
    print(f"\n[deck] Searching for hero matching '{args.hero}'...")

    # Build filtered list so we can navigate forward and back
    matches = [
        d for d in all_decks
        if hero_query in d.get("hero", "").lower()
        or hero_query in d["name"].lower()
    ]

    if not matches:
        print(f"  No decks matching '{args.hero}'.")
        return

    matches.sort(key=lambda d: d.get("win_rate", 0), reverse=True)
    print(f"  Found {len(matches)} matching deck(s) (sorted by win rate).\n")

    idx = 0
    while 0 <= idx < len(matches):
        deck = matches[idx]

        wr = deck.get("win_rate", 0)
        fmt = deck.get("format", "")
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{len(matches)}] Deck: {deck['name']}")
        if deck.get("hero"):
            print(f"  Hero: {deck['hero']}")
        print(f"  Win rate: {wr:.0f}%")
        if fmt:
            print(f"  Format: {fmt}")
        print(f"  URL:  {deck['url']}")
        print(f"{'='*60}")

        while True:
            prev_hint = " / [p]rev" if idx > 0 else ""
            choice = input(f"\n  [i]nspect / [d]ownload / [s]kip{prev_hint} / [q]uit? ").strip().lower()

            if choice == "q":
                print("Bye!")
                return

            if choice == "s":
                idx += 1
                break

            if choice == "p" and idx > 0:
                idx -= 1
                break

            if choice in ("i", "d"):
                print("  Fetching deck details...", flush=True)
                detail = _fetch_deck_detail(deck["url"])

                if choice == "i":
                    _print_deck_detail(detail, set_map)
                    continue  # ask again after inspection

                # Download
                if not detail["hero_arena"]:
                    print("  [ERROR] No hero/equipment found — cannot download.")
                    continue

                content, warnings = _build_deck_file(detail, set_map)
                for w in warnings:
                    print(f"  [WARN] {w}")

                suggested = _generate_filename(detail, set_map)
                name = input(f"  Save as Assets/[{suggested}].txt: ").strip()
                if not name:
                    name = suggested

                assets_dir = Path(__file__).resolve().parents[2] / "Assets"
                out_path = assets_dir / f"{name}.txt"
                out_path.write_text(content)
                print(f"  Saved to {out_path}")
                print(f"  Use with: --p1-deck {name} or --p2-deck {name}")
                idx += 1
                break

    print("\n[deck] Done — no more matching decks.")


if __name__ == "__main__":
    main()
