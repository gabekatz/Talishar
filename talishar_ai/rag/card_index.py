"""
card_index.py — Card knowledge base backed by lucisearch.

Indexes all ~4,633 cards from GeneratedCardDictionaries.php with their
stats, keywords, and rate-system valuations.  Provides direct lookup by
card ID and filtered search by card type, keywords, and hero synergy.

Build the index once offline via ``scripts/build_card_index.py``, then
load it at game time for O(1) card lookups and filtered queries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .index_store import IndexStore, SearchResult

# ---------------------------------------------------------------------------
# Paths relative to this file (talishar_ai/rag/)
# ---------------------------------------------------------------------------

_DEFAULT_PHP_PATH = (
    Path(__file__).parent.parent.parent
    / "GeneratedCode"
    / "GeneratedCardDictionaries.php"
)
_DEFAULT_VOCAB_PATH = Path(__file__).parent.parent / "card_vocab.json"
_DEFAULT_METADATA_PATH = Path(__file__).parent.parent / "card_metadata.json"
_DEFAULT_INDEX_DIR = Path(__file__).parent.parent / "indices"

# ---------------------------------------------------------------------------
# Rate system constants (FaB baseline: 3 = on-rate)
# ---------------------------------------------------------------------------

_RATE_BASELINE = 3.0

# Keyword bonuses for attack_value calculation
_KEYWORD_ATTACK_BONUS: dict[str, float] = {
    "go_again": 1.0,
    "dominate": 1.5,
    "piercing": 0.5,
    "intimidate": 0.5,
    "overpower": 0.5,
    "crush": 0.5,
}

# Block willingness penalties for durability keywords
_BLOCK_KEYWORD_FLOOR: dict[str, float] = {
    "battleworn": 0.6,
    "blade_break": 0.5,
    "temper": 0.5,
}


# ---------------------------------------------------------------------------
# CardDocument
# ---------------------------------------------------------------------------


@dataclass
class CardDocument:
    """Schema for one card in the index."""

    card_id: str  # e.g. "command_and_conquer_red"
    name: str  # e.g. "Command and Conquer"
    card_type: str  # AA, A, AR, DR, E, W, I, T, C, Event
    subtype: str  # class/talent info
    cost: int
    power: int
    defense: int
    pitch: int  # 0, 1, 2, or 3

    # Keywords as a list of strings
    keywords: list[str] = field(default_factory=list)

    # Rate-system valuations
    attack_value: float = 0.0
    block_willingness: float = 0.5
    arsenal_value: float = 0.0
    best_use_value: float = 0.0

    # Card class and talent for hero synergy
    card_class: str = ""  # "Ninja", "Warrior", etc.
    card_talent: str = ""  # "Shadow", "Ice", etc.

    # Go-again flag (separate from keywords for quick access)
    has_go_again: bool = False

    # Rules / ability text from Fabrary (e.g. "Instant - Destroy this: ...")
    functional_text: str = ""

    # Human-readable description for LLM prompts
    description: str = ""

    def to_index_doc(self) -> dict[str, Any]:
        """Convert to flat dict for luci indexing."""
        return {
            "_id": self.card_id,
            "card_id": self.card_id,
            "name": self.name,
            "description": self.description,
            "functional_text": self.functional_text,
            "card_type": self.card_type,
            "subtype": self.subtype,
            "cost": float(self.cost),
            "power": float(self.power),
            "defense": float(self.defense),
            "pitch": float(self.pitch),
            "attack_value": self.attack_value,
            "block_willingness": self.block_willingness,
            "arsenal_value": self.arsenal_value,
            "best_use_value": self.best_use_value,
            "card_class": self.card_class,
            "card_talent": self.card_talent,
            "has_go_again": "true" if self.has_go_again else "false",
            # Store keywords as comma-separated string (luci doesn't support arrays)
            "keywords_csv": ",".join(self.keywords) if self.keywords else "",
        }

    @classmethod
    def from_index_doc(cls, doc: dict[str, Any]) -> "CardDocument":
        """Reconstruct from luci document (flat _source dict)."""
        # luci returns _source as a flat dict; handle both nested and flat
        src = doc.get("_source", doc)
        return cls(
            card_id=src.get("card_id", doc.get("_id", "")),
            name=src.get("name", ""),
            card_type=src.get("card_type", ""),
            subtype=src.get("subtype", ""),
            cost=int(src.get("cost", 0)),
            power=int(src.get("power", 0)),
            defense=int(src.get("defense", 0)),
            pitch=int(src.get("pitch", 0)),
            keywords=[
                k for k in src.get("keywords_csv", "").split(",") if k
            ],
            attack_value=float(src.get("attack_value", 0.0)),
            block_willingness=float(src.get("block_willingness", 0.5)),
            arsenal_value=float(src.get("arsenal_value", 0.0)),
            best_use_value=float(src.get("best_use_value", 0.0)),
            card_class=src.get("card_class", ""),
            card_talent=src.get("card_talent", ""),
            has_go_again=src.get("has_go_again") == "true",
            functional_text=src.get("functional_text", ""),
            description=src.get("description", ""),
        )


# ---------------------------------------------------------------------------
# PHP parser
# ---------------------------------------------------------------------------

# Regex for match statement entries: "card_id" => value,
_MATCH_STR_RE = re.compile(r'"([a-z][a-z0-9_]*)"  *=> *"([^"]*)"')
_MATCH_INT_RE = re.compile(r'"([a-z][a-z0-9_]*)"  *=> *(-?\d+)')
_MATCH_BOOL_RE = re.compile(r'"([a-z][a-z0-9_]*)"  *=> *(true|false)')

# Function name extractor
_FUNC_RE = re.compile(r"^function (Generated\w+)\(")


def _parse_php_dictionaries(
    php_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Parse GeneratedCardDictionaries.php into a dict of card_id -> properties.

    Single-pass parser: reads the file once, identifies each Generated*
    function, and extracts all card_id => value mappings from its match
    statement.

    Returns
    -------
    dict mapping card_id to a dict of properties, e.g.:
    {
        "command_and_conquer_red": {
            "type": "AA",
            "power": 6,
            "defense": 3,
            "name": "Command and Conquer",
            "cost": 6,
            "pitch": 1,
            "go_again": False,
            "class": "Generic",
            "talent": "",
            "keywords": ["dominate"],
            ...
        }
    }
    """
    cards: dict[str, dict[str, Any]] = {}
    current_func = ""

    # Map function names to property keys and value types
    func_map: dict[str, tuple[str, str]] = {
        "GeneratedCardType": ("type", "str"),
        "GeneratedPowerValue": ("power", "int"),
        "GeneratedBlockValue": ("defense", "int"),
        "GeneratedCardName": ("name", "str"),
        "GeneratedPitchValue": ("pitch", "int"),
        "GeneratedCardCost": ("cost", "int"),
        "GeneratedCardSubtype": ("subtype", "str"),
        "GeneratedCardClass": ("class", "str"),
        "GeneratedCardTalent": ("talent", "str"),
        "GeneratedGoAgain": ("go_again", "bool"),
        "GeneratedIs1H": ("is_1h", "bool"),
        "GeneratedCharacterHealth": ("health", "int"),
        "GeneratedCharacterIntellect": ("intellect", "int"),
    }

    # Keyword functions: GeneratedHasX -> keyword name
    keyword_funcs: dict[str, str] = {
        "GeneratedHasAmbush": "ambush",
        "GeneratedHasBattleworn": "battleworn",
        "GeneratedHasBladeBreak": "blade_break",
        "GeneratedHasBoost": "boost",
        "GeneratedHasChannel": "channel",
        "GeneratedHasCharge": "charge",
        "GeneratedHasClash": "clash",
        "GeneratedHasCloaked": "cloaked",
        "GeneratedHasCombo": "combo",
        "GeneratedHasCrush": "crush",
        "GeneratedHasDecompose": "decompose",
        "GeneratedHasDominate": "dominate",
        "GeneratedHasEphemeral": "ephemeral",
        "GeneratedHasFreeze": "freeze",
        "GeneratedHasGalvanize": "galvanize",
        "GeneratedHasHeave": "heave",
        "GeneratedHasHeavy": "heavy",
        "GeneratedHasIntimidate": "intimidate",
        "GeneratedHasLegendary": "legendary",
        "GeneratedHasMeld": "meld",
        "GeneratedHasOpt": "opt",
        "GeneratedHasOverpower": "overpower",
        "GeneratedHasPhantasm": "phantasm",
        "GeneratedHasPiercing": "piercing",
        "GeneratedHasReprise": "reprise",
        "GeneratedHasSpectra": "spectra",
        "GeneratedHasSpellvoid": "spellvoid",
        "GeneratedHasStealth": "stealth",
        "GeneratedHasSurge": "surge",
        "GeneratedHasTemper": "temper",
        "GeneratedHasWard": "ward",
    }

    with open(php_path) as f:
        for line in f:
            stripped = line.strip()

            # Detect function boundaries
            func_match = _FUNC_RE.match(stripped)
            if func_match:
                current_func = func_match.group(1)
                continue

            # Skip non-match lines
            if "=>" not in stripped or stripped.startswith("default"):
                continue

            # -- Standard property functions --
            if current_func in func_map:
                prop_name, val_type = func_map[current_func]

                if val_type == "str":
                    m = _MATCH_STR_RE.match(stripped)
                    if m:
                        card_id, val = m.group(1), m.group(2)
                        cards.setdefault(card_id, {})[prop_name] = val

                elif val_type == "int":
                    m = _MATCH_INT_RE.match(stripped)
                    if m:
                        card_id, val = m.group(1), int(m.group(2))
                        cards.setdefault(card_id, {})[prop_name] = val

                elif val_type == "bool":
                    m = _MATCH_BOOL_RE.match(stripped)
                    if m:
                        card_id = m.group(1)
                        val = m.group(2) == "true"
                        cards.setdefault(card_id, {})[prop_name] = val

            # -- Keyword functions --
            elif current_func in keyword_funcs:
                keyword_name = keyword_funcs[current_func]
                m = _MATCH_BOOL_RE.match(stripped)
                if m and m.group(2) == "true":
                    card_id = m.group(1)
                    entry = cards.setdefault(card_id, {})
                    entry.setdefault("keywords", []).append(keyword_name)

    return cards


# ---------------------------------------------------------------------------
# Rate-system valuation
# ---------------------------------------------------------------------------


def _compute_attack_value(power: int, keywords: list[str]) -> float:
    """
    Keyword-adjusted attack value.

    Base = power/2 (a 6-power card has base attack value 3 = on-rate).
    Keywords like go_again, dominate, piercing add bonuses.
    """
    base = power / 2.0
    bonus = sum(_KEYWORD_ATTACK_BONUS.get(k, 0) for k in keywords)
    return base + bonus


def _compute_block_willingness(
    defense: int,
    card_type: str,
    keywords: list[str],
) -> float:
    """
    How willing are we to use this card for blocking (0-1)?

    Equipment and weapons: low willingness (permanent resources).
    Cards with durability keywords: lower willingness.
    High-value attack cards: lower willingness (save for offense).
    """
    if card_type in ("E", "W"):
        # Equipment/weapons: rarely want to block with these
        return 0.1

    # Base: higher defense = more willing to block
    base = min(defense / 4.0, 1.0)

    # Durability keyword penalty
    for kw, floor in _BLOCK_KEYWORD_FLOOR.items():
        if kw in keywords:
            base = min(base, floor)

    return base


def _compute_arsenal_value(
    card_type: str,
    power: int,
    cost: int,
    keywords: list[str],
    has_go_again: bool,
) -> float:
    """
    Value of this card sitting in arsenal.

    Cards that can only pitch/block (no play effect) are dead in arsenal.
    Attack actions, instants, and reactions have arsenal value proportional
    to their play strength.
    """
    # Equipment and weapons: can't play from arsenal in most cases
    if card_type in ("E", "W", "C", "T"):
        return 0.0

    # Defense reactions: situational value from arsenal
    if card_type == "DR":
        return 2.0

    # Attack actions: value scales with power + keyword bonuses
    if card_type == "AA":
        return _compute_attack_value(power, keywords)

    # Action cards, instants, reactions: moderate arsenal value
    if card_type in ("A", "I", "AR"):
        return max(2.0, _compute_attack_value(power, keywords))

    return 1.0


def _build_description(
    name: str,
    card_type: str,
    cost: int,
    power: int,
    defense: int,
    pitch: int,
    keywords: list[str],
    has_go_again: bool,
    card_class: str,
    card_talent: str,
    functional_text: str = "",
) -> str:
    """Build a human-readable description for LLM prompts."""
    parts = [name + "."]

    # Type description
    type_names = {
        "AA": "Attack Action",
        "A": "Action",
        "AR": "Attack Reaction",
        "DR": "Defense Reaction",
        "I": "Instant",
        "E": "Equipment",
        "W": "Weapon",
        "C": "Hero",
        "T": "Token",
        "Event": "Event",
    }
    type_str = type_names.get(card_type, card_type)
    parts.append(f"{type_str}.")

    # Stats
    stat_parts = []
    if cost > 0:
        stat_parts.append(f"Cost {cost}")
    if power > 0:
        stat_parts.append(f"Power {power}")
    if defense > 0:
        stat_parts.append(f"Defense {defense}")
    if pitch > 0:
        color = {1: "Red", 2: "Yellow", 3: "Blue"}.get(pitch, f"Pitch {pitch}")
        stat_parts.append(color)
    if stat_parts:
        parts.append(", ".join(stat_parts) + ".")

    # Class/talent
    if card_class:
        parts.append(f"Class: {card_class}.")
    if card_talent:
        parts.append(f"Talent: {card_talent}.")

    # Functional text (rules text from Fabrary) — most important for LLM
    if functional_text:
        parts.append(f"Rules: {functional_text}")

    # Keywords (only if no functional text, since rules text usually covers them)
    if not functional_text:
        kw_list = list(keywords)
        if has_go_again and "go_again" not in kw_list:
            kw_list.append("go_again")
        if kw_list:
            parts.append("Keywords: " + ", ".join(kw_list) + ".")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# CardIndex
# ---------------------------------------------------------------------------


class CardIndex:
    """
    Manages the card knowledge index.

    Build once offline via ``CardIndex.build()``, then instantiate at
    game time via ``CardIndex(persist_dir)`` for fast lookups.
    """

    def __init__(
        self,
        store: IndexStore | None = None,
        persist_dir: str | Path | None = None,
    ) -> None:
        if store is not None:
            self._store = store
        else:
            pd = Path(persist_dir) if persist_dir else _DEFAULT_INDEX_DIR
            self._store = IndexStore("cards", persist_dir=pd)

        # In-memory cache for O(1) lookup by card_id
        self._cache: dict[str, CardDocument] = {}
        self._cache_loaded = False

    def _ensure_cache(self) -> None:
        """Lazy-load the full card set into memory for fast lookup."""
        if self._cache_loaded:
            return
        # Fetch all cards via a match_all search with large size
        results = self._store.search(query=None, top_k=10000)
        for r in results:
            cd = CardDocument.from_index_doc(r.document)
            self._cache[cd.card_id] = cd
        self._cache_loaded = True

    @classmethod
    def build(
        cls,
        php_path: str | Path | None = None,
        persist_dir: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> "CardIndex":
        """
        Build the card index from GeneratedCardDictionaries.php.

        Parses all card stats, keywords, and type info, then computes
        rate-system valuations and indexes everything in lucisearch.

        If card_metadata.json is available (enriched with Fabrary data),
        functional_text (card rules/abilities) is included so the LLM
        knows what each card actually does.

        Parameters
        ----------
        php_path:
            Path to GeneratedCardDictionaries.php.
        persist_dir:
            Where to store the lucisearch index on disk.
        metadata_path:
            Path to card_metadata.json (enriched with Fabrary data).

        Returns
        -------
        A ready-to-use CardIndex instance.
        """
        src = Path(php_path) if php_path else _DEFAULT_PHP_PATH
        pd = Path(persist_dir) if persist_dir else _DEFAULT_INDEX_DIR
        meta_path = Path(metadata_path) if metadata_path else _DEFAULT_METADATA_PATH

        print(f"[CardIndex] Parsing {src}...")
        raw_cards = _parse_php_dictionaries(src)
        print(f"[CardIndex] Parsed {len(raw_cards):,} cards")

        # Load enriched metadata (functional text from Fabrary)
        metadata: dict[str, dict] = {}
        if meta_path.exists():
            print(f"[CardIndex] Loading metadata from {meta_path}...")
            try:
                content = meta_path.read_text()
                # Fix known JSON issues in card_metadata.json
                import re as _re
                content = _re.sub(r'"([^"]*)",([a-zA-Z])', r'"\1",', content)
                content = _re.sub(r",(\s*[}\]])", r"\1", content)
                metadata = json.loads(content)
                has_func = sum(
                    1 for m in metadata.values() if m.get("functional_text")
                )
                print(f"[CardIndex]   {len(metadata):,} cards, "
                      f"{has_func:,} with functional text")
            except Exception as e:
                print(f"[CardIndex]   WARNING: Failed to load metadata: {e}")
        else:
            print(f"[CardIndex] No metadata file at {meta_path} — "
                  f"descriptions will lack ability text")

        # Build CardDocuments with computed valuations
        card_docs: list[CardDocument] = []
        for card_id, props in raw_cards.items():
            meta = metadata.get(card_id, {})

            # Use PHP parser as primary, enriched metadata as fallback
            card_type = props.get("type", "") or meta.get("type", "")
            power = props.get("power", 0)
            defense = props.get("defense", 0)
            cost = props.get("cost", 0)
            pitch = props.get("pitch", 0)
            card_class = props.get("class", "") or meta.get("class", "")
            card_talent = props.get("talent", "") or meta.get("talent", "")
            name = props.get("name", "") or meta.get("name", "") or card_id.replace("_", " ").title()

            # Merge keywords from PHP and metadata (metadata is more complete)
            php_keywords = props.get("keywords", [])
            meta_keywords = [k.lower().replace(" ", "_") for k in meta.get("keywords", [])]
            # Use metadata keywords if available (authoritative), else PHP
            keywords = meta_keywords if meta_keywords else php_keywords

            has_go_again = props.get("go_again", False) or "go_again" in keywords
            if has_go_again and "go_again" not in keywords:
                keywords = keywords + ["go_again"]

            # Pull functional text from enriched metadata
            functional_text = meta.get("functional_text", "")

            attack_value = _compute_attack_value(power, keywords)
            block_will = _compute_block_willingness(defense, card_type, keywords)
            arsenal_val = _compute_arsenal_value(
                card_type, power, cost, keywords, has_go_again
            )
            best_use = max(attack_value, arsenal_val)

            description = _build_description(
                name,
                card_type,
                cost,
                power,
                defense,
                pitch,
                keywords,
                has_go_again,
                card_class,
                card_talent,
                functional_text=functional_text,
            )

            card_docs.append(
                CardDocument(
                    card_id=card_id,
                    name=name,
                    card_type=card_type,
                    subtype=props.get("subtype", ""),
                    cost=cost,
                    power=power,
                    defense=defense,
                    pitch=pitch,
                    keywords=keywords,
                    attack_value=attack_value,
                    block_willingness=block_will,
                    arsenal_value=arsenal_val,
                    best_use_value=best_use,
                    card_class=card_class,
                    card_talent=card_talent,
                    has_go_again=has_go_again,
                    functional_text=functional_text,
                    description=description,
                )
            )

        # Delete old index so we rebuild with the new schema
        old_index = pd / "cards.luci"
        if old_index.exists():
            old_index.unlink()
            print(f"[CardIndex] Removed old index at {old_index}")

        # Index in lucisearch
        store = IndexStore("cards", persist_dir=pd)
        store.add_documents([cd.to_index_doc() for cd in card_docs])
        store.commit()

        print(f"[CardIndex] Indexed {len(card_docs):,} cards → {pd}")

        idx = cls(store=store)
        # Pre-populate cache
        for cd in card_docs:
            idx._cache[cd.card_id] = cd
        idx._cache_loaded = True
        return idx

    # ------------------------------------------------------------------
    # Lookup operations
    # ------------------------------------------------------------------

    def lookup(self, card_ids: list[str]) -> list[CardDocument]:
        """
        Direct lookup for cards by ID.  O(1) per card from cache.

        Unknown IDs are silently skipped.
        """
        self._ensure_cache()
        return [self._cache[cid] for cid in card_ids if cid in self._cache]

    def lookup_one(self, card_id: str) -> CardDocument | None:
        """Look up a single card by ID. Returns None if not found."""
        self._ensure_cache()
        return self._cache.get(card_id)

    # ------------------------------------------------------------------
    # Search operations
    # ------------------------------------------------------------------

    def search_by_type(
        self,
        card_type: str,
        top_k: int = 20,
    ) -> list[CardDocument]:
        """Find all cards of a given type (AA, DR, E, etc.)."""
        results = self._store.query_by_metadata(
            {"card_type": card_type}, top_k=top_k
        )
        return [CardDocument.from_index_doc(r.document) for r in results]

    def search_by_keywords(
        self,
        keywords: list[str],
        card_type: str | None = None,
        top_k: int = 20,
    ) -> list[CardDocument]:
        """
        Find cards that have ALL specified keywords.

        Optionally filter by card type.  Uses in-memory cache since
        luci doesn't natively support list-valued term filters.
        """
        self._ensure_cache()
        kw_set = set(keywords)
        candidates = [
            c for c in self._cache.values()
            if kw_set.issubset(set(c.keywords))
        ]
        if card_type:
            candidates = [c for c in candidates if c.card_type == card_type]
        candidates.sort(key=lambda c: c.best_use_value, reverse=True)
        return candidates[:top_k]

    def search_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[CardDocument]:
        """
        Full-text search across card descriptions.

        Useful for natural-language queries like "go again ninja attack".
        """
        results = self._store.search(query, top_k=top_k)
        return [CardDocument.from_index_doc(r.document) for r in results]

    def search_high_value(
        self,
        card_type: str | None = None,
        card_class: str | None = None,
        min_value: float = _RATE_BASELINE,
        top_k: int = 20,
    ) -> list[CardDocument]:
        """
        Find high best_use_value cards, optionally filtered by type/class.
        """
        self._ensure_cache()
        candidates = self._cache.values()

        if card_type:
            candidates = [c for c in candidates if c.card_type == card_type]
        if card_class:
            candidates = [c for c in candidates if c.card_class == card_class]

        above_rate = [c for c in candidates if c.best_use_value >= min_value]
        above_rate.sort(key=lambda c: c.best_use_value, reverse=True)
        return above_rate[:top_k]

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------

    def all_cards(self) -> list[CardDocument]:
        """Return all indexed cards (loads full cache if needed)."""
        self._ensure_cache()
        return list(self._cache.values())

    def card_count(self) -> int:
        """Number of cards in the index."""
        return self._store.count()

    def __repr__(self) -> str:
        return f"CardIndex(cards={self.card_count()})"
