"""
hand_evaluator.py — Domain-knowledge hand evaluation for Flesh and Blood.

This module encodes FaB strategic knowledge that the RL model struggles
to learn from reward signals alone.  The evaluator takes a hand of cards
+ game context and returns structured plans (attack lines, defense
allocations, arsenal picks) that can be used as:

  1. Extra observation features (injected into features.py)
  2. Strategy mask inputs (inform env.py which actions are dominated)
  3. Imitation targets (penalise the model for deviating from the plan)

All card data comes from the GetAIState JSON — the same dicts the model
already sees.  No external lookups needed.

Usage:
    from talishar_ai.hand_evaluator import HandEvaluator

    evaluator = HandEvaluator()
    plan = evaluator.evaluate(state)
    plan.attack.total_damage   # best attack line damage
    plan.attack.sequence       # ordered list of cards to play
    plan.defense.cards_to_block # which cards to block with
    plan.arsenal_pick           # card to arsenal, if any
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Card:
    """Simplified card view extracted from GetAIState JSON."""
    card_id: str = ""
    cost: int = 0
    power: int = 0
    defense: int = 0
    pitch: int = 0       # 1=red, 2=yellow, 3=blue
    card_type: str = ""  # AA, I, A, E, DR, R, T, C
    has_go_again: bool = False
    keywords: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @staticmethod
    def from_move(move: dict) -> Card:
        """Build from a legalMoves[] entry."""
        stats = move.get("stats") or {}
        return Card(
            card_id=move.get("cardID", "") or "",
            cost=int(stats.get("cost", 0) or 0),
            power=int(stats.get("power", 0) or 0),
            defense=int(stats.get("defense", 0) or 0),
            pitch=int(stats.get("pitch", 0) or 0),
            card_type=(stats.get("type") or "").split(",")[0].strip(),
            raw=move,
        )

    @staticmethod
    def from_state_card(card: dict) -> Card:
        """Build from a myState zone card entry (hand, equipment, etc.)."""
        stats = card.get("stats") or {}
        return Card(
            card_id=card.get("cardID", "") or "",
            cost=int(stats.get("cost", 0) or 0),
            power=int(stats.get("power", 0) or 0),
            defense=int(stats.get("defense", 0) or 0),
            pitch=int(stats.get("pitch", 0) or 0),
            card_type=(stats.get("type") or "").split(",")[0].strip(),
            raw=card,
        )


@dataclass
class AttackPlan:
    """Optimal attack sequence for the current hand."""
    sequence: list[Card] = field(default_factory=list)
    pitch_cards: list[Card] = field(default_factory=list)
    total_damage: int = 0
    total_pitch_cost: int = 0
    uses_weapons: bool = False
    ends_with_no_go_again: bool = False  # turn-ender as last action

    # Cards left over after the attack plan (available to block/arsenal)
    surplus: list[Card] = field(default_factory=list)


@dataclass
class DefensePlan:
    """Optimal blocking allocation."""
    cards_to_block: list[Card] = field(default_factory=list)
    total_block_value: int = 0
    damage_after_block: int = 0
    cards_held_back: list[Card] = field(default_factory=list)


@dataclass
class HandPlan:
    """Complete hand evaluation."""
    attack: AttackPlan = field(default_factory=AttackPlan)
    defense: DefensePlan = field(default_factory=DefensePlan)
    arsenal_pick: Card | None = None


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------

class HandEvaluator:
    """
    Stateless hand evaluator.  All context comes from the state dict.

    Fill in the methods below with your FaB knowledge.  Each method
    has comments explaining what it should compute and why.
    """

    def evaluate(self, state: dict[str, Any]) -> HandPlan:
        """Full hand evaluation given the current game state."""
        my = state.get("myState", {})
        phase = (state.get("phase") or {}).get("turnPhase", "")
        turn_no = int(state.get("turnNumber", 0) or 0)

        hand = [Card.from_state_card(c) for c in my.get("hand", [])]
        equipment = [Card.from_state_card(c) for c in my.get("equipment", [])]

        cc = state.get("combatChain") or {}
        incoming_damage = (
            int(cc.get("totalPower", 0) or 0)
            - int(cc.get("totalDefense", 0) or 0)
        )

        plan = HandPlan()

        if phase in ("M", "A"):
            plan.attack = self.best_attack_line(hand, equipment, state)
            plan.arsenal_pick = self.best_arsenal_pick(
                plan.attack.surplus, state
            )
        elif phase == "D":
            plan.defense = self.best_defense(
                hand, equipment, incoming_damage, turn_no
            )

        return plan

    # ------------------------------------------------------------------
    # ATTACK PLANNING — fill this in
    # ------------------------------------------------------------------

    def best_attack_line(
        self,
        hand: list[Card],
        equipment: list[Card],
        state: dict[str, Any],
    ) -> AttackPlan:
        """Compute the highest-damage attack sequence from this hand.

        FaB attack sequencing rules you'll want to encode:
        - Weapons (Kodachi) have go-again and cost 0 — play them FIRST
        - Cards that grant go-again should come before cards that don't
        - The last attack should be your biggest (turn-ender, no go-again)
        - Pitch blue cards to pay for costs before committing them to attack
        - Reactions/pumps (Razor Reflex etc.) boost the current attack

        Parameters
        ----------
        hand : cards currently in hand
        equipment : available equipment (weapons, etc.)
        state : full game state for context (resources, auras, etc.)

        Returns
        -------
        AttackPlan with ordered sequence, pitch allocation, damage total,
        and surplus cards left over.
        """
        plan = AttackPlan()

        # TODO: Your logic here.  Example structure:
        #
        # 1. Separate hand into: attacks, non-attacks, pitch fodder
        # 2. Identify available weapons from equipment
        # 3. Build sequence: weapons first (free go-again), then
        #    go-again attacks, then biggest non-go-again as closer
        # 4. Allocate pitch cards to cover costs
        # 5. Remaining cards go to surplus

        return plan

    # ------------------------------------------------------------------
    # DEFENSE PLANNING — fill this in
    # ------------------------------------------------------------------

    def best_defense(
        self,
        hand: list[Card],
        equipment: list[Card],
        incoming_damage: int,
        turn_no: int,
    ) -> DefensePlan:
        """Compute optimal blocking allocation.

        FaB defense rules you'll want to encode:
        - Turn 0: block with ALL hand cards (they're free, redraw to intellect)
        - Overblocking is fine on turn 0 (no downside)
        - Later turns: block efficiently (minimize cards used for damage prevented)
        - Never block with high-utility equipment when hand cards suffice
        - Consider on-hit effects: if opponent has on-hit triggers,
          it's worth overblocking to prevent them
        - Hold back cards you want for your next attack turn (if not turn 0)

        Parameters
        ----------
        hand : cards currently in hand
        equipment : equipment that could block
        incoming_damage : unblocked damage remaining
        turn_no : current turn number

        Returns
        -------
        DefensePlan with cards to block, total block value, remaining damage.
        """
        plan = DefensePlan()

        # TODO: Your logic here.  Example structure:
        #
        # Turn 0:
        #   Block with every hand card. Period.
        #   plan.cards_to_block = [c for c in hand if c.defense > 0]
        #
        # Later turns:
        #   1. Sort hand by "block efficiency" (defense value vs offensive value)
        #   2. Add cards until incoming_damage is covered
        #   3. Consider on-hit: if active, try to fully block even at cost
        #   4. Never use equipment unless hand can't cover it

        return plan

    # ------------------------------------------------------------------
    # ARSENAL PLANNING — fill this in
    # ------------------------------------------------------------------

    def best_arsenal_pick(
        self,
        surplus: list[Card],
        state: dict[str, Any],
    ) -> Card | None:
        """Pick the best card to arsenal from leftover cards.

        FaB arsenal rules:
        - Always arsenal something if you have a surplus card and empty arsenal
        - Prefer high-power attacks (save them for next turn's closer)
        - Cards with on-hit effects are great arsenal picks
        - Blue cards are poor arsenal picks (low power, better as pitch)
        - Reactions are decent arsenal picks (can be used defensively or offensively)

        Parameters
        ----------
        surplus : cards left over after attack plan
        state : full game state (check arsenal slot availability)

        Returns
        -------
        Best card to arsenal, or None if nothing is worth saving.
        """
        # TODO: Your logic here.  Example:
        #
        # arsenal = state.get("myState", {}).get("arsenal", [])
        # if len(arsenal) > 0 or not surplus:
        #     return None
        #
        # # Pick the highest-power attack from surplus
        # attacks = [c for c in surplus if c.card_type == "AA"]
        # if attacks:
        #     return max(attacks, key=lambda c: c.power)
        #
        # # Otherwise pick highest value card
        # return max(surplus, key=lambda c: c.power + c.defense)

        return None
