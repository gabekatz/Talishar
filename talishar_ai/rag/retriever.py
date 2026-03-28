"""
retriever.py — Phase-aware context assembly from card and combo indices.

The Retriever is the bridge between raw GetAIState JSON and the consumers
(LLM agent, PPO feature encoder, BC recorder).  It examines the current
game phase and dispatches to phase-specific methods that query the card
and combo indices for relevant information.

Different phases need different information:
- Main/Action: best attack combos, hand metadata, lethal lines
- Defense: incoming attack severity, block values, equipment break costs
- Arsenal: carry-over value comparison for each hand card
- Turn 0: free-hand blocking, equipment preservation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .card_index import CardDocument, CardIndex


# ---------------------------------------------------------------------------
# RetrievalContext — standardised output consumed by all three consumers
# ---------------------------------------------------------------------------


@dataclass
class RetrievalContext:
    """
    Everything retrieved for one decision point.

    Consumers interpret this differently:
    - LLM agent: formats as structured prompt text
    - PPO feature consumer: encodes numerics into a feature vector
    - BC recorder: stores alongside the chosen action
    """

    # Current game phase
    phase: str  # "M", "A", "D", "B", "P", "ARS", "OVER", etc.
    turn_number: int

    # My cards with full metadata
    hand_cards: list[CardDocument] = field(default_factory=list)
    arsenal_cards: list[CardDocument] = field(default_factory=list)
    equipment: list[CardDocument] = field(default_factory=list)

    # Opponent's visible cards with metadata
    opponent_equipment: list[CardDocument] = field(default_factory=list)

    # Phase-specific retrievals
    best_combos: list[dict[str, Any]] = field(default_factory=list)
    counter_options: list[CardDocument] = field(default_factory=list)

    # Game scalars
    my_health: int = 0
    opp_health: int = 0
    resources: int = 0
    action_points: int = 0
    my_deck_count: int = 0
    opp_deck_count: int = 0
    opp_hand_count: int = 0

    # Combat chain state
    combat_chain_power: int = 0
    combat_chain_defense: int = 0
    combat_chain_keywords: list[str] = field(default_factory=list)
    attacking_card: CardDocument | None = None

    # Strategic signals (pre-computed for consumers)
    can_threaten_lethal: bool = False
    incoming_lethal: bool = False
    is_turn_zero: bool = False

    # Hero info
    hero_id: str = ""
    opponent_hero_id: str = ""

    # Aggregated hand stats
    total_hand_power: int = 0
    total_hand_defense: int = 0
    total_hand_pitch: int = 0
    avg_hand_value: float = 0.0
    best_arsenal_candidate: CardDocument | None = None


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    Phase-aware context assembly from card and combo indices.

    Parameters
    ----------
    card_index:
        Built CardIndex with all card metadata.
    combo_index:
        Optional ComboIndex for attack sequence planning (Phase 5).
        When None, combo-related fields in RetrievalContext are empty.
    """

    def __init__(
        self,
        card_index: CardIndex,
        combo_index: Any | None = None,
    ) -> None:
        self._cards = card_index
        self._combos = combo_index

    def retrieve(self, state: dict[str, Any]) -> RetrievalContext:
        """
        Main entry point.  Examines phase and dispatches to the
        appropriate phase-specific method.

        Parameters
        ----------
        state:
            Raw GetAIState JSON response (the same dict the env sees).
        """
        phase_info = state.get("phase") or {}
        phase = phase_info.get("turnPhase", "")
        turn_number = int(state.get("turnNumber", 0) or 0)

        # Build base context (shared across all phases)
        ctx = self._build_base_context(state, phase, turn_number)

        # Detect turn 0
        if turn_number == 0:
            ctx.is_turn_zero = True
            self._enrich_turn_zero(ctx, state)
        elif phase in ("M", "A"):
            self._enrich_main_phase(ctx, state)
        elif phase == "D":
            self._enrich_defense_phase(ctx, state)
        elif phase == "ARS":
            self._enrich_arsenal_phase(ctx, state)
        else:
            # Fallback: main phase enrichment for other phases
            self._enrich_main_phase(ctx, state)

        return ctx

    # ------------------------------------------------------------------
    # Base context (shared across all phases)
    # ------------------------------------------------------------------

    def _build_base_context(
        self,
        state: dict[str, Any],
        phase: str,
        turn_number: int,
    ) -> RetrievalContext:
        """Build the common fields that every phase needs."""
        my = state.get("myState", {})
        opp = state.get("theirState", {})
        cc = state.get("combatChain", {}) or {}

        def _i(v: Any) -> int:
            return int(v or 0)

        # Look up card metadata for cards in hand/arsenal/equipment
        hand_ids = [c.get("cardID", "") for c in my.get("hand", [])]
        arsenal_ids = [c.get("cardID", "") for c in my.get("arsenal", [])]
        equip_ids = [c.get("cardID", "") for c in my.get("equipment", [])]
        opp_equip_ids = [c.get("cardID", "") for c in opp.get("equipment", [])]

        hand_cards = self._cards.lookup(hand_ids)
        arsenal_cards = self._cards.lookup(arsenal_ids)
        equipment = self._cards.lookup(equip_ids)
        opp_equipment = self._cards.lookup(opp_equip_ids)

        # Combat chain keywords
        cc_keywords = []
        if cc.get("goAgain"):
            cc_keywords.append("go_again")
        if cc.get("dominate"):
            cc_keywords.append("dominate")
        if cc.get("piercing"):
            cc_keywords.append("piercing")

        # Attacking card metadata
        attacking_card = None
        attack_id = cc.get("attackingCard", {})
        if isinstance(attack_id, dict):
            atk_cid = attack_id.get("cardID", "")
        else:
            atk_cid = str(attack_id) if attack_id else ""
        if atk_cid:
            attacking_card = self._cards.lookup_one(atk_cid)

        # Aggregated hand stats
        total_power = sum(c.power for c in hand_cards)
        total_defense = sum(c.defense for c in hand_cards)
        total_pitch = sum(c.pitch for c in hand_cards)
        avg_value = (
            sum(c.best_use_value for c in hand_cards) / len(hand_cards)
            if hand_cards
            else 0.0
        )

        # Best arsenal candidate
        best_arsenal = None
        if hand_cards:
            playable = [c for c in hand_cards if c.arsenal_value > 0]
            if playable:
                best_arsenal = max(playable, key=lambda c: c.arsenal_value)

        my_health = _i(my.get("health", 0))
        opp_health = _i(opp.get("health", 0))
        cc_power = _i(cc.get("totalPower", 0))

        # Hero IDs from character zone
        hero_id = ""
        chars = my.get("character", my.get("hero", []))
        if isinstance(chars, list) and chars:
            hero_id = chars[0].get("cardID", "") if isinstance(chars[0], dict) else ""

        opp_hero_id = ""
        opp_chars = opp.get("character", opp.get("hero", []))
        if isinstance(opp_chars, list) and opp_chars:
            opp_hero_id = (
                opp_chars[0].get("cardID", "") if isinstance(opp_chars[0], dict) else ""
            )

        return RetrievalContext(
            phase=phase,
            turn_number=turn_number,
            hand_cards=hand_cards,
            arsenal_cards=arsenal_cards,
            equipment=equipment,
            opponent_equipment=opp_equipment,
            my_health=my_health,
            opp_health=opp_health,
            resources=_i(my.get("resources", 0)),
            action_points=_i(my.get("ap", 0)),
            my_deck_count=_i(my.get("deckCount", 0)),
            opp_deck_count=_i(opp.get("deckCount", 0)),
            opp_hand_count=_i(opp.get("handCount", 0)),
            combat_chain_power=cc_power,
            combat_chain_defense=_i(cc.get("totalDefense", 0)),
            combat_chain_keywords=cc_keywords,
            attacking_card=attacking_card,
            can_threaten_lethal=(
                opp_health > 0 and total_power >= opp_health
            ),
            incoming_lethal=(my_health > 0 and cc_power >= my_health),
            hero_id=hero_id,
            opponent_hero_id=opp_hero_id,
            total_hand_power=total_power,
            total_hand_defense=total_defense,
            total_hand_pitch=total_pitch,
            avg_hand_value=avg_value,
            best_arsenal_candidate=best_arsenal,
        )

    # ------------------------------------------------------------------
    # Phase-specific enrichment
    # ------------------------------------------------------------------

    def _enrich_main_phase(
        self, ctx: RetrievalContext, state: dict[str, Any]
    ) -> None:
        """
        Main/Action phase: find achievable combos, lethal lines,
        and resource planning info.
        """
        if self._combos is not None:
            hand_ids = [c.card_id for c in ctx.hand_cards]
            ctx.best_combos = self._combos.find_achievable(
                hand_card_ids=hand_ids,
                available_resources=ctx.resources + ctx.total_hand_pitch,
                hero_id=ctx.hero_id,
                top_k=5,
            )

    def _enrich_defense_phase(
        self, ctx: RetrievalContext, state: dict[str, Any]
    ) -> None:
        """
        Defense phase: assess incoming attack, compute block assignments,
        and evaluate equipment break costs.

        Adds counter_options: hand cards sorted by block efficiency.
        """
        # Sort hand cards by block willingness × defense value
        # (high willingness + high defense = block with these first)
        blockers = sorted(
            ctx.hand_cards,
            key=lambda c: c.block_willingness * c.defense,
            reverse=True,
        )
        ctx.counter_options = blockers

    def _enrich_arsenal_phase(
        self, ctx: RetrievalContext, state: dict[str, Any]
    ) -> None:
        """
        Arsenal phase: rank hand cards by arsenal_value.
        Cards with arsenal_value=0 are dead weight in arsenal.
        """
        # Already have best_arsenal_candidate from base context.
        # Sort hand cards by arsenal value for the LLM to see.
        ctx.hand_cards = sorted(
            ctx.hand_cards,
            key=lambda c: c.arsenal_value,
            reverse=True,
        )

    def _enrich_turn_zero(
        self, ctx: RetrievalContext, state: dict[str, Any]
    ) -> None:
        """
        Turn 0 special case.

        Hand cards are "free" to block (redrawn at end of turn).
        Equipment must be preserved at all costs.
        Offensive value is low unless attack line exceeds expected
        block value (8-12 points).
        """
        # Recalculate block willingness: all hand cards are free to block
        for card in ctx.hand_cards:
            card.block_willingness = 1.0

        # Equipment: explicitly set to never block
        for card in ctx.equipment:
            card.block_willingness = 0.0

    # ------------------------------------------------------------------
    # Prompt rendering (for LLM consumers)
    # ------------------------------------------------------------------

    @staticmethod
    def render_for_llm(
        ctx: RetrievalContext,
        legal_moves: list[dict[str, Any]],
    ) -> str:
        """
        Render a RetrievalContext + legal moves as a structured prompt
        string for the LLM agent.
        """
        lines: list[str] = []

        # Header
        phase_names = {
            "M": "Main Phase",
            "A": "Action Phase",
            "D": "Defense Phase",
            "B": "Begin Phase",
            "ARS": "Arsenal Phase",
            "P": "Priority",
        }
        phase_name = phase_names.get(ctx.phase, ctx.phase)
        lines.append(f"PHASE: {phase_name} (Turn {ctx.turn_number})")
        if ctx.is_turn_zero:
            lines.append(
                "*** TURN 0: Hand cards are FREE to block (redrawn). "
                "NEVER break equipment. ***"
            )
        lines.append(
            f"MY HEALTH: {ctx.my_health} | OPPONENT HEALTH: {ctx.opp_health}"
        )
        potential_resources = ctx.resources + ctx.total_hand_pitch
        lines.append(
            f"RESOURCES: {ctx.resources} (floating) | "
            f"POTENTIAL: {potential_resources} (pitch cards from hand to generate more) | "
            f"ACTION POINTS: {ctx.action_points}"
        )
        lines.append(
            f"MY DECK: {ctx.my_deck_count} | OPP DECK: {ctx.opp_deck_count} "
            f"| OPP HAND: {ctx.opp_hand_count}"
        )
        lines.append("")

        # Hand cards
        if ctx.hand_cards:
            lines.append(f"MY HAND ({len(ctx.hand_cards)} cards):")
            for i, c in enumerate(ctx.hand_cards, 1):
                kw_str = ", ".join(c.keywords) if c.keywords else "none"
                card_line = (
                    f"  {i}. {c.name} ({c.card_id}) — "
                    f"Type: {c.card_type}, Cost: {c.cost}, "
                    f"Power: {c.power}, Defense: {c.defense}, "
                    f"Pitch: {c.pitch}. "
                    f"Keywords: {kw_str}. "
                    f"Value: {c.best_use_value:.1f}. "
                    f"Block willingness: {c.block_willingness:.1f}"
                )
                if c.functional_text:
                    card_line += f"\n     Ability: {c.functional_text}"
                lines.append(card_line)
            lines.append(
                f"  >> Total hand: power={ctx.total_hand_power}, "
                f"defense={ctx.total_hand_defense}, pitch={ctx.total_hand_pitch}, "
                f"avg value={ctx.avg_hand_value:.1f}"
            )
            lines.append("")

        # Arsenal
        if ctx.arsenal_cards:
            lines.append("MY ARSENAL:")
            for c in ctx.arsenal_cards:
                lines.append(f"  - {c.name} ({c.card_id}) — {c.description}")
            lines.append("")

        # Equipment
        if ctx.equipment:
            lines.append("MY EQUIPMENT:")
            for c in ctx.equipment:
                equip_line = (
                    f"  - {c.name} ({c.card_id}) — Def: {c.defense}. "
                    f"Keywords: {', '.join(c.keywords) if c.keywords else 'none'}"
                )
                if c.functional_text:
                    equip_line += f"\n    Ability: {c.functional_text}"
                lines.append(equip_line)
            lines.append("")

        # Combat chain (defense phase)
        if ctx.phase == "D" and ctx.combat_chain_power > 0:
            lines.append("INCOMING ATTACK:")
            if ctx.attacking_card:
                lines.append(
                    f"  {ctx.attacking_card.name} — "
                    f"Power: {ctx.combat_chain_power}, "
                    f"Keywords: {', '.join(ctx.combat_chain_keywords) or 'none'}"
                )
            else:
                lines.append(f"  Total power: {ctx.combat_chain_power}")
            if ctx.incoming_lethal:
                lines.append("  *** LETHAL INCOMING — MUST BLOCK OR DIE ***")
            lines.append("")

        # Best combos (main phase)
        if ctx.best_combos:
            lines.append("BEST ATTACK LINES:")
            for i, combo in enumerate(ctx.best_combos, 1):
                cards_str = " -> ".join(combo.get("cards", []))
                lines.append(
                    f"  {chr(64 + i)}) {cards_str} = "
                    f"{combo.get('total_damage', 0)} damage"
                    f"{' (go-again chain)' if combo.get('has_go_again_chain') else ''}"
                    f"{' DOMINATE' if combo.get('dominate') else ''}"
                )
            lines.append("")

        # Strategic signals
        signals = []
        if ctx.can_threaten_lethal:
            signals.append("CAN THREATEN LETHAL this turn")
        if ctx.incoming_lethal:
            signals.append("INCOMING LETHAL if unblocked")
        if ctx.best_arsenal_candidate and ctx.phase in ("M", "A"):
            signals.append(
                f"Best arsenal candidate: {ctx.best_arsenal_candidate.name} "
                f"(value {ctx.best_arsenal_candidate.arsenal_value:.1f})"
            )
        if signals:
            lines.append("STRATEGIC SIGNALS: " + " | ".join(signals))
            lines.append("")

        # Legal moves
        lines.append(f"LEGAL MOVES ({len(legal_moves)}):")
        for i, move in enumerate(legal_moves):
            desc = move.get("description", "")
            move_type = move.get("type", "")
            card_id = move.get("cardID", "")
            label = desc or f"{move_type}: {card_id}" if card_id else move_type
            lines.append(f"  [{i}] {label}")

        return "\n".join(lines)
