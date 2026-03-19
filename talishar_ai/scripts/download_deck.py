"""
scripts/download_deck.py — Interactive CLI to browse and download decks from
fabrary.net for use in Talishar AI training.

Usage
-----
uv run python -m scripts.download_deck --hero "Ira"
uv run python -m scripts.download_deck --hero "Kayo"
uv run python -m scripts.download_deck              # lists all heroes

The tool scrapes https://fabrary.net/most-played-decks, lets you browse by
hero, inspect deck contents, and download decks to Assets/ in the format
expected by CreateTrainingGame.php.
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

def _fetch_deck_list(max_pages: int = 6) -> list[dict]:
    """
    Fetch the most-played decks listing from fabrary.net.

    Returns a list of dicts with keys: name, href, url.
    """
    from playwright.sync_api import sync_playwright

    decks: list[dict] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for pg in range(1, max_pages + 1):
            url = "https://fabrary.net/most-played-decks"
            if pg > 1:
                url += f"?page={pg}"
            print(f"[deck] Fetching page {pg}...", end=" ", flush=True)
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Wait for deck links to appear (ads keep network active forever)
            page.wait_for_selector('a[href*="/decks/"]', timeout=15000)

            links = page.query_selector_all('a[href*="/decks/"]')
            page_decks = []
            for link in links:
                href = link.get_attribute("href") or ""
                name = link.inner_text().split("\n")[0].strip()
                if not href or not name:
                    continue

                # Extract hero from hero image in parent container.
                # The image URL contains the hero slug:
                #   content.fabrary.net/heroes/kayo-armed-and-dangerous.webp
                hero = ""
                try:
                    hero_info = link.evaluate('''el => {
                        let p = el;
                        for (let i = 0; i < 5; i++) {
                            p = p.parentElement;
                            if (!p) break;
                            const imgs = p.querySelectorAll("img");
                            for (const img of imgs) {
                                const src = img.src || "";
                                if (src.includes("/heroes/")) {
                                    const slug = src.split("/heroes/")[1].replace(".webp", "");
                                    return slug;
                                }
                            }
                        }
                        return "";
                    }''')
                    hero = hero_info.replace("-", " ").title() if hero_info else ""
                except Exception:
                    pass

                page_decks.append({
                    "name": name,
                    "hero": hero,
                    "href": href,
                    "url": f"https://fabrary.net{href}",
                })
            print(f"{len(page_decks)} decks")
            decks.extend(page_decks)

            if not page_decks:
                break

        browser.close()

    return decks


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
    args = parser.parse_args()

    set_map = _load_set_id_map()

    print("[deck] Fetching most-played decks from fabrary.net...")
    all_decks = _fetch_deck_list(max_pages=args.pages)

    if not all_decks:
        print("[ERROR] No decks found.")
        return

    if args.hero is None:
        print(f"\n  Found {len(all_decks)} decks. Use --hero to filter.\n")
        for i, d in enumerate(all_decks):
            hero_tag = f" [{d['hero']}]" if d.get("hero") else ""
            print(f"  {i+1:3d}. {d['name']}{hero_tag}")
        return

    hero_query = args.hero.lower()
    print(f"\n[deck] Searching for hero matching '{args.hero}'...")

    for deck in all_decks:
        # Match against hero name (extracted from image) or deck name
        hero_name = deck.get("hero", "").lower()
        deck_name = deck["name"].lower()
        if hero_query not in hero_name and hero_query not in deck_name:
            continue

        print(f"\n{'='*60}")
        print(f"  Deck: {deck['name']}")
        if deck.get("hero"):
            print(f"  Hero: {deck['hero']}")
        print(f"  URL:  {deck['url']}")
        print(f"{'='*60}")

        while True:
            choice = input("\n  [i]nspect / [d]ownload / [s]kip / [q]uit? ").strip().lower()

            if choice == "q":
                print("Bye!")
                return

            if choice == "s":
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

                # Generate filename from hero card name
                hero_id = _resolve_card_id(
                    detail["hero_arena"][0]["code"],
                    detail["hero_arena"][0]["name"],
                    set_map,
                )
                default_name = hero_id or detail["hero_arena"][0]["name"].replace(" ", "_")
                # Clean up for filename: capitalize words
                default_name = default_name.replace(",", "").replace("'", "")
                suggested = "".join(
                    w.capitalize() for w in default_name.split("_")
                )

                name = input(f"  Save as Assets/[{suggested}].txt: ").strip()
                if not name:
                    name = suggested

                assets_dir = Path(__file__).resolve().parents[2] / "Assets"
                out_path = assets_dir / f"{name}.txt"
                out_path.write_text(content)
                print(f"  Saved to {out_path}")
                print(f"  Use with: --p1-deck {name} or --p2-deck {name}")
                break

    print("\n[deck] Done — no more matching decks.")


if __name__ == "__main__":
    main()
