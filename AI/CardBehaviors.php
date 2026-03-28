<?php

/**
 * CardBehaviors.php — Dynamic card priority computation
 *
 * Priorities are derived from card stats (power, block, pitch, cost, type)
 * so the AI works for any deck without a per-card table.
 *
 * Priority array layout (8 values):
 *   [0] Block priority          — 0 = won't block; 0.1–0.9 = will block
 *   [1] Action priority         — 0 = won't play; 0.1–0.9 = will play
 *   [2] Arsenal-action priority — same scale
 *   [3] Reaction priority       — 0 = won't react; 0.1–0.9 = will react
 *   [4] Arsenal-reaction prio   — same scale
 *   [5] Pitch priority          — 0 = won't pitch; higher = prefer to pitch
 *   [6] Send-to-arsenal prio    — 0 = won't; higher = prefer
 *   [7] Permanent/equip ability — 0 = won't activate; higher = will
 */

/**
 * Entry point.  Returns priority array for $cardID when used by $heroID.
 * State-dependent cards (Kodachi, etc.) override the dynamic result.
 */
function GetCardBehavior($cardID, $heroID)
{
    // --- State-based overrides (must read game state dynamically) -----------
    switch ($cardID) {
        case "harmonized_kodachi":
            return ComputeKodachiPriority(2);
        case "flic_flak_blue":
            return ComputeFlichFlakBluePriority(2);
        case "fai_rising_rebellion":
            return ComputeFaiHeroPriority(2);
        case "art_of_war_yellow":
            return ComputeArtOfWarPriority(2);
        case "snapdragon_scalers":
            return ComputeSnapdragonPriority(2);
    }

    return ComputeCardBehavior($cardID);
}

/**
 * Derive the priority array from card stats.
 * Works for any card without explicit table entries.
 */
function ComputeCardBehavior($cardID)
{
    $pitchVal  = max(0, (int)PitchValue($cardID));
    $blockVal  = (int)BlockValue($cardID);        // -2 / -1 = can't block
    $powerVal  = max(0, (int)PowerValue($cardID, 2));
    $cost      = max(0, (int)CardCost($cardID));
    $abCost    = max(0, (int)AbilityCost($cardID));
    $cardType  = CardType($cardID);
    $abilType  = GetAbilityType($cardID, -1, "CHAR");

    $isEquip   = ($cardType === "E" || $cardType === "W" || $cardType === "C");
    $isDR      = ($cardType === "DR");
    $isAA      = ($cardType === "AA");
    $isInst    = (str_contains($abilType, "I") || $cardType === "I");

    // ---- Equipment / hero cards: handled by state-based overrides or left 0 --
    // Don't auto-activate arbitrary equipment — only cards with explicit
    // Compute* overrides (e.g. Kodachi) should have non-zero ability priority.
    if ($isEquip) {
        return [0, 0, 0, 0, 0, 0, 0, 0];
    }

    // ---- Pitch priority -------------------------------------------------------
    // DRs are low-action-value → prefer pitching them.
    // AAs have attack value → slightly lower pitch priority.
    $pitchPrio = 0.0;
    if ($pitchVal > 0) {
        if ($isDR) {
            // Defense reactions are pitch fodder — high pitch priority
            $pitchPrio = 0.5 + $pitchVal * 1.0;   // red=1.5, yellow=2.5, blue=3.5
        } else {
            // Attack/action cards — pitch only if necessary
            $pitchPrio = 0.3 + $pitchVal * 0.7;   // red=1.0, yellow=1.7, blue=2.4
        }
    }

    // ---- Block priority -------------------------------------------------------
    $blockPrio = 0.0;
    if ($blockVal > 0) {
        if ($isDR) {
            // Defense reactions: free to block, always excellent
            $blockPrio = min(0.9, 0.5 + $blockVal * 0.15);
        } else {
            // Other cards: efficiency = block / (cost + 1)
            $blockPrio = min(0.85, $blockVal / ($cost + 1) * 0.15 + 0.1);
        }
    }

    // ---- Action priority (play from hand/arsenal) ----------------------------
    $actionPrio = 0.0;
    if ($isDR) {
        $actionPrio = 0.0;  // DRs are not played as main actions
    } elseif ($isAA && $powerVal > 0) {
        // Higher power, lower cost = higher priority
        $actionPrio = min(0.9, 0.15 + $powerVal / 10.0 + ($cost == 0 ? 0.25 : 0.0));
    } elseif ($isInst) {
        $actionPrio = min(0.6, 0.2 + $powerVal / 15.0);
    }

    // ---- Reaction priority (play during opponent's attack) -------------------
    $reactPrio = 0.0;
    if ($isDR && $blockVal > 0) {
        $reactPrio = min(0.9, 0.5 + $blockVal * 0.15);
    } elseif ($isInst) {
        $reactPrio = min(0.5, 0.1 + $powerVal / 15.0);
    }

    // ---- Send-to-arsenal priority --------------------------------------------
    // Prefer to arsenal cards that are good to save for later (high action prio,
    // low pitch value to avoid wasting pitch fodder).
    $arsenalPrio = $actionPrio > 0.5 ? $actionPrio * 0.8 : 0.0;

    return [
        $blockPrio,    // [0] Block
        $actionPrio,   // [1] Action
        $actionPrio,   // [2] Arsenal action
        $reactPrio,    // [3] Reaction
        $reactPrio,    // [4] Arsenal reaction
        $pitchPrio,    // [5] Pitch
        $arsenalPrio,  // [6] Arsenal
        0.0,           // [7] Permanent/equipment ability (overrides only)
    ];
}

// ---------------------------------------------------------------------------
// State-based computed priorities
// These must read live game state to make context-aware decisions.
// ---------------------------------------------------------------------------

function ComputeKodachiPriority($playerID)
{
    $resources  = &GetResources($playerID);
    $blueCount  = SearchCount(SearchHand($playerID, "pitch", 3));

    // Activate if we have blue cards to pitch (pays the cost while keeping
    // attack going) or if we already have excess resources.
    $shouldActivate = ($blueCount > 0 || $resources[0] > 1);

    $prio = $shouldActivate ? 0.95 : 0.1;
    return [0, $prio, 0, 0, 0, 0, 0, $prio];
}

function ComputeFlichFlakBluePriority($playerID)
{
    $blueCount = SearchCount(SearchHand($playerID, "pitch", 3));

    // React if we have more than one blue (keep one for attacking)
    $reactPrio = $blueCount > 1 ? 0.9 : 0.1;
    return [0.6, 0.0, 0.0, $reactPrio, $blueCount > 1 ? $reactPrio : 0, 2.9, 0.8, 0];
}

function ComputeFaiHeroPriority($playerID)
{
    $resources   = &GetResources($playerID);
    $chainLinks  = NumDraconicChainLinks();

    $prio = ($chainLinks >= 3 || ($resources[0] > $chainLinks && $chainLinks > 0)) ? 1.0 : 0.0;
    return [0, $prio, 0, 0, 0, 0, 0, $prio];
}

function ComputeArtOfWarPriority($playerID)
{
    $blueCount = SearchCount(SearchHand($playerID, "pitch", 3));

    $playPrio = (!ArsenalEmpty($playerID) && $blueCount > 0) ? 1.0 : 0.0;
    return [0, $playPrio, $playPrio, 0, 0, 2.5, 1.0, 0];
}

/**
 * Compatibility shim for AIDebugger.php which iterated over a per-hero table.
 * Now that behavior is dynamic, this returns an empty array — the debugger
 * functions that called it are development-only and not invoked during play.
 */
function GetCardBehaviorForHero($heroID)
{
    return [];
}

function ComputeSnapdragonPriority($playerID)
{
    $hasGoAgain = DoesAttackHaveGoAgain();
    $resources  = &GetResources($playerID);
    $hand       = &GetHand($playerID);

    $shouldActivate = !$hasGoAgain && (count($hand) > 0 || $resources[0] > 0);
    return [0, 0, 0, 0, 0, 0, 0, $shouldActivate ? 0.85 : 0];
}
