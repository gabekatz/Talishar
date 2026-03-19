"""
card_vocab.py — Card ID vocabulary for learned embeddings.

Builds and persists a mapping of  card_id_string → integer_index  extracted
from GeneratedCardDictionaries.php.  Index 0 is always PAD (empty slot or
unknown card), so the embedding layer can use ``padding_idx=0``.

Usage
-----
# Build once (writes card_vocab.json next to this file)
>>> CardVocab.build(php_path="path/to/GeneratedCardDictionaries.php")

# Load at runtime
>>> vocab = CardVocab()        # loads card_vocab.json from default location
>>> vocab.encode("ira")        # → int
>>> vocab.size                 # total entries including PAD
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Default paths relative to this file (talishar_ai/)
_DEFAULT_VOCAB_PATH = Path(__file__).parent / "card_vocab.json"
_DEFAULT_PHP_PATH   = (
    Path(__file__).parent.parent / "GeneratedCode" / "GeneratedCardDictionaries.php"
)

# Regex matching lines like: "some_card_id_blue" => "TYPE",
# Card IDs are always lowercase letters, digits, and underscores.
_CARD_ID_RE = re.compile(r'^"([a-z][a-z0-9_]*)" =>')


class CardVocab:
    """
    Integer vocabulary for Flesh and Blood card IDs.

    Index 0 = PAD (used for empty card slots and unknown IDs).
    Indices 1 … vocab.size-1 are the canonical card IDs sorted alphabetically.

    Parameters
    ----------
    path:
        Path to a pre-built card_vocab.json file.  If omitted, the default
        path ``talishar_ai/card_vocab.json`` is used.
    """

    PAD_ID = 0

    def __init__(self, path: str | Path | None = None) -> None:
        vocab_path = Path(path) if path else _DEFAULT_VOCAB_PATH
        with open(vocab_path) as f:
            self._map: dict[str, int] = json.load(f)
        self.size = len(self._map)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, card_id: str) -> int:
        """
        Return the integer index for *card_id*.

        Unknown IDs (including empty strings) map to PAD_ID (0).
        """
        return self._map.get(card_id, self.PAD_ID)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        php_path: str | Path | None = None,
        out_path: str | Path | None = None,
    ) -> "CardVocab":
        """
        Parse GeneratedCardDictionaries.php, extract every unique card ID,
        and write the vocabulary to *out_path* as JSON.

        Parameters
        ----------
        php_path:
            Path to GeneratedCardDictionaries.php.  Defaults to
            ``../../GeneratedCode/GeneratedCardDictionaries.php`` relative to
            this file.
        out_path:
            Where to write card_vocab.json.  Defaults to the same directory
            as this file.

        Returns
        -------
        The freshly built CardVocab instance.
        """
        src = Path(php_path) if php_path else _DEFAULT_PHP_PATH
        dst = Path(out_path) if out_path else _DEFAULT_VOCAB_PATH

        card_ids: set[str] = set()
        with open(src) as f:
            for line in f:
                m = _CARD_ID_RE.match(line.strip())
                if m:
                    card_ids.add(m.group(1))

        # Deterministic order: sort alphabetically so the vocab is stable
        # across re-builds (new cards always get new indices at the end).
        vocab: dict[str, int] = {"PAD": cls.PAD_ID}
        for idx, cid in enumerate(sorted(card_ids), start=1):
            vocab[cid] = idx

        with open(dst, "w") as f:
            json.dump(vocab, f, indent=2)

        print(f"[CardVocab] Built vocab: {len(vocab):,} entries → {dst}")
        return cls(dst)

    @classmethod
    def load_or_build(
        cls,
        vocab_path: str | Path | None = None,
        php_path:   str | Path | None = None,
    ) -> "CardVocab":
        """
        Load vocab from *vocab_path* if it exists, otherwise build it first.

        Convenience method for use in training scripts.
        """
        vp = Path(vocab_path) if vocab_path else _DEFAULT_VOCAB_PATH
        if not vp.exists():
            print(f"[CardVocab] {vp} not found — building from PHP source…")
            return cls.build(php_path=php_path, out_path=vp)
        return cls(vp)

    def __repr__(self) -> str:
        return f"CardVocab(size={self.size})"


# ---------------------------------------------------------------------------
# Script: python -m talishar_ai.card_vocab
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build card_vocab.json from GeneratedCardDictionaries.php")
    p.add_argument("--php-path", default=str(_DEFAULT_PHP_PATH))
    p.add_argument("--out-path", default=str(_DEFAULT_VOCAB_PATH))
    args = p.parse_args()
    CardVocab.build(php_path=args.php_path, out_path=args.out_path)
