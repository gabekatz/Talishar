<?php

/**
 * EncounterPriorityValues.php - AI card priority definitions
 * 
 * This file determines what priority each card gets in each phase.
 * Instead of directly editing this file for each card, use CardBehaviors.php
 * to define card priorities in a cleaner, more maintainable format.
 * 
 * PRIORITY RANGES:
 * [0] = Block Priority
 *   0 = won't block
 *   0.1-0.9 = block if lethal/efficient
 *   10.1-10.9 = computed (highest -> 0.1-0.9, rest -> 2.1-2.9)
 * 
 * [1] = Action Priority
 *   0 = won't play
 *   0.1-0.9 = will play (higher first)
 *   10.1-10.9 = computed (highest -> 0, rest -> 0.1-0.9)
 * 
 * [2] = Arsenal Action Priority
 *   0 = won't play
 *   0.1-0.9 = will play (higher first)
 * 
 * [3] = Reaction Priority
 *   0 = won't play
 *   0.1-0.9 = will play (higher first)
 *   10.1-10.9 = computed
 * 
 * [4] = Arsenal Reaction Priority
 *   0 = won't play
 *   0.1-0.9 = will play (higher first)
 * 
 * [5] = Pitch Priority
 *   0 = won't pitch
 *   0.1+ = pitch value (higher = prefer to pitch)
 * 
 * [6] = Arsenal Priority
 *   0 = won't arsenal
 *   0.1-0.9 = will arsenal (higher first)
 * 
 * [7] = Permanent Ability Priority
 *   0 = won't activate
 *   0.1-0.9 = will activate (higher first)
 * 
 * HOW TO ADD/EDIT CARDS:
 * Card priorities are now computed dynamically from card stats (power, block,
 * pitch, cost, card type) in CardBehaviors.php::ComputeCardBehavior().
 * To override specific cards (e.g. state-dependent logic), add a case to the
 * switch at the top of GetCardBehavior() and implement a Compute* function.
 */

// Include supporting files
include_once "CardBehaviors.php";
include_once "AIHelpers.php";
include_once "AIDebugger.php";

/**
 * Get priority value for a card in a specific context
 * 
 * @param string $cardID - Card identifier
 * @param string $heroID - Hero identifier
 * @param int $type - Priority type index (0-7)
 * 
 * @return float - Priority value (0-0.9, or 10.1-10.9 for computed)
 */
function GetPriority($cardID, $heroID, $type)
{
  // Get the card behavior from the new system
  return GetCardBehavior($cardID, $heroID)[$type] ?? 0;
}

/**
 * Check if encounter should block on first turn
 * Can override per-hero if you want specific encounter behavior
 */
function EncounterBlocksFirstTurn($heroID)
{
  switch($heroID) {
    default:
      return true; 
  }
}

?>