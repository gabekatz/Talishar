<?php

/**
 * GetAIState.php — AI state extraction endpoint
 *
 * Returns a structured JSON game state optimized for AI consumption,
 * including enriched card data and pre-computed legal moves.
 *
 * GET params:
 *   gameName  — numeric game ID
 *   playerID  — 1 or 2
 *   authKey   — player's auth UUID
 *
 * Response shape:
 * {
 *   gameID, playerID, turnNumber, turnPlayer, havePriority,
 *   phase: { turnPhase, caption },
 *   myState:   { health, resources, ap, deckCount, hand[], arsenal[], equipment[], ... },
 *   theirState: { health, handCount, deckCount, equipment[], ... },
 *   combatChain: { attackingCard, totalPower, totalDefense, goAgain, dominate, ... } | null,
 *   stack: { target, layerContents[] } | null,
 *   pendingDecision: { type, context, options[] } | null,
 *   legalMoves: [
 *     { id, type, mode, cardID?, zone?, params: {mode, cardID?, buttonInput?, ...}, description }
 *   ]
 * }
 *
 * To submit a legal move, POST the `params` object plus gameName/playerID/authKey
 * to SubmitAIAction.php, OR issue a GET to ProcessInput.php with the same params
 * as URL query parameters.
 */

include 'WriteLog.php';
include 'GameLogic.php';
include 'GameTerms.php';
include 'HostFiles/Redirector.php';
include 'Libraries/SHMOPLibraries.php';
include 'Libraries/StatFunctions.php';
include 'Libraries/UILibraries.php';
include 'Libraries/PlayerSettings.php';
include 'Libraries/CacheLibraries.php';
include 'Libraries/HTTPLibraries.php';
require_once 'Libraries/CoreLibraries.php';
include_once 'includes/dbh.inc.php';
include_once 'BuildGameState.php';
include_once 'BuildPlayerInputPopup.php';

SetHeaders();
header('Content-Type: application/json; charset=utf-8');

$gameName = $_GET['gameName'] ?? '';
if (!IsGameNameValid($gameName)) {
    echo json_encode(['error' => 'Invalid game name.']);
    exit;
}

$playerID = intval($_GET['playerID'] ?? 0);
if ($playerID !== 1 && $playerID !== 2) {
    echo json_encode(['error' => 'playerID must be 1 or 2.']);
    exit;
}

$authKey = $_GET['authKey'] ?? '';

if (!file_exists('./Games/' . $gameName . '/GameFile.txt')) {
    echo json_encode(['error' => 'Game not found.']);
    exit;
}

// Release session lock before any file I/O
if (session_status() === PHP_SESSION_ACTIVE) {
    session_write_close();
}

$sessionData = [
    'userLoggedIn'     => false,
    'userName'         => null,
    'isPvtVoidPatron'  => false,
    'patreonCampaigns' => [],
    'friendList'       => [],
];

$gs = BuildGameStateResponse($gameName, $playerID, $authKey, $sessionData, false);

if (is_string($gs)) {
    echo json_encode(['error' => $gs]);
    exit;
}

echo json_encode(BuildAIState($gs, $playerID, $gameName));
exit;

// ---------------------------------------------------------------------------
// State assembly
// ---------------------------------------------------------------------------

function BuildAIState(stdClass $gs, int $playerID, string $gameName): array
{
    $havePriority = $gs->havePriority ?? false;
    $legalMoves   = $havePriority ? CollectLegalMoves($gs) : [];

    return [
        'gameID'          => (string)($gs->initialLoad->gameGUID ?? $gameName),
        'playerID'        => $playerID,
        'turnNumber'      => $gs->turnNo ?? 0,
        'turnPlayer'      => $gs->turnPlayer ?? 0,
        'firstPlayer'     => $gs->firstPlayer ?? 0,
        'havePriority'    => $havePriority,
        'phase'           => [
            'turnPhase' => $gs->turnPhase->turnPhase ?? '',
            'layer'     => $gs->turnPhase->layer ?? null,
            'caption'   => $gs->turnPhase->caption ?? '',
        ],
        'myState'         => BuildMyState($gs),
        'theirState'      => BuildTheirState($gs),
        'combatChain'     => BuildCombatChain($gs),
        'stack'           => BuildStack($gs),
        'pendingDecision' => BuildPendingDecision($gs),
        'legalMoves'      => $legalMoves,
    ];
}

// ---------------------------------------------------------------------------
// Legal move collection
// ---------------------------------------------------------------------------

function CollectLegalMoves(stdClass $gs): array
{
    $moves  = [];
    $nextID = 0;

    // ---- Phases that need dedicated move generation -------------------------
    $turnPhaseStr = $gs->turnPhase->turnPhase ?? '';

    // INPUTCARDNAME: player must name a card (e.g. Censor hit effect).
    // Return only the mode-30 move — passing doesn't resolve this phase.
    if ($turnPhaseStr === 'INPUTCARDNAME') {
        $handCards = $gs->playerHand ?? [];
        $namedCard = !empty($handCards) ? ($handCards[0]->cardNumber ?? 'Enlightened_Strike') : 'Enlightened_Strike';
        return [[
            'id'          => 0,
            'type'        => 'INPUT_CARD_NAME',
            'mode'        => 30,
            'params'      => ['mode' => 30, 'buttonInput' => $namedCard],
            'description' => 'Name a card (' . $namedCard . ')',
        ]];
    }

    // PDECK: pitch-to-deck ordering --------------------------------
    // During PDECK, the player must choose which pitched card to put on deck
    // bottom (mode 6).  Normal zone scanning doesn't cover this.
    if ($turnPhaseStr === 'PDECK') {
        foreach ($gs->playerPitch ?? [] as $idx => $pitchCard) {
            $cardID = $pitchCard->cardNumber ?? '';
            if ($cardID === '' || $cardID === 'CARDBACK') continue;
            $moves[] = [
                'id'          => $nextID++,
                'type'        => 'PITCH_TO_DECK',
                'mode'        => 6,
                'cardID'      => $cardID,
                'zone'        => 'PITCH',
                'stats'       => CardStats($cardID),
                'params'      => ['mode' => 6, 'cardID' => $cardID],
                'description' => "Put on deck bottom: $cardID",
            ];
        }
        // PDECK only needs pitch-to-deck moves — skip normal zone scanning
        if (!empty($moves)) return $moves;
    }

    // ---- Zone cards --------------------------------------------------------
    // Each entry: [zone label, cards array]
    $zoneSets = [
        ['HAND',       $gs->playerHand       ?? []],
        ['ARSENAL',    $gs->playerArse        ?? []],
        ['EQUIPMENT',  $gs->playerEquipment  ?? []],
        ['AURAS',      $gs->playerAuras       ?? []],
        ['ITEMS',      $gs->playerItems       ?? []],
        ['ALLIES',     $gs->playerAllies      ?? []],
        ['PERMANENTS', $gs->playerPermanents  ?? []],
        ['DISCARD',    $gs->playerDiscard     ?? []],
        ['BANISH',     $gs->playerBanish      ?? []],
    ];

    // Deck card (Dash heroes can play off the top)
    if (isset($gs->playerDeckCard) && ($gs->playerDeckCard->action ?? 0) !== 0) {
        $moves[] = CardMove($nextID++, $gs->playerDeckCard, 'DECK');
    }

    // Pre-compute affordability bounds.
    // Playing any card may trigger P phase if resources[0] < cost.
    // If the remaining hand cards can't cover the cost, the P phase becomes
    // unresolvable (no cards left to pitch).  Filter those moves out here.
    $currentResources = intval($gs->playerPitchCount ?? 0);
    $handSize = count((array)($gs->playerHand ?? []));
    $maxAffordable = $currentResources + $handSize * 3; // each card pitches at most 3

    // Total pitch available across all hand cards (for per-card affordability check)
    $totalHandPitch = 0;
    foreach ((array)($gs->playerHand ?? []) as $hCard) {
        $totalHandPitch += max(0, (int)PitchValue($hCard->cardNumber ?? ''));
    }

    foreach ($zoneSets as [$zone, $cards]) {
        foreach ($cards as $card) {
            if (($card->action ?? 0) !== 0) {
                if ($zone === 'EQUIPMENT') {
                    $cost = AbilityCost($card->cardNumber ?? '');
                    if ($cost > $maxAffordable) continue;
                }
                if ($zone === 'HAND' && intval($card->action ?? 0) === 27 && $turnPhaseStr !== 'P') {
                    // Affordability check only during main phase play (not P phase pitch).
                    // In P phase, hand cards also use action=27 but are being pitched,
                    // so the card's play cost is irrelevant.
                    $cardNum = $card->cardNumber ?? '';
                    $cost = max(0, (int)CardCost($cardNum));
                    if ($cost > 0) {
                        // Playing this card removes it from hand — can we pitch enough
                        // from the *remaining* cards to cover what resources don't cover?
                        $cardPitch = max(0, (int)PitchValue($cardNum));
                        $affordableFromRemaining = $currentResources + ($totalHandPitch - $cardPitch);
                        if ($cost > $affordableFromRemaining) continue;
                    }
                }
                // Arsenal affordability: playing from arsenal (mode 5) can also
                // trigger P phase.  Filter if the hand can't cover the cost.
                if ($zone === 'ARSENAL') {
                    $cardNum = $card->cardNumber ?? '';
                    $cost = max(0, (int)CardCost($cardNum));
                    if ($cost > $currentResources + $totalHandPitch) continue;
                }
                $moves[] = CardMove($nextID++, $card, $zone);
            }
        }
    }

    // ---- Opponent-zone plays (opp arsenal, opp banish) ---------------------
    $oppZones = [
        ['OPP_ARSENAL', $gs->opponentArse   ?? []],
        ['OPP_BANISH',  $gs->opponentBanish ?? []],
    ];
    foreach ($oppZones as [$zone, $cards]) {
        foreach ($cards as $card) {
            if (($card->action ?? 0) !== 0) {
                $moves[] = CardMove($nextID++, $card, $zone);
            }
        }
    }

    // ---- Active combat chain abilities -------------------------------------
    $chainLink = $gs->activeChainLink ?? null;
    if ($chainLink) {
        $chainCards = array_merge(
            [$chainLink->attackingCard ?? null],
            $chainLink->reactions ?? []
        );
        foreach ($chainCards as $card) {
            if ($card && ($card->action ?? 0) !== 0) {
                $moves[] = CardMove($nextID++, $card, 'COMBAT_CHAIN');
            }
        }
    }

    // ---- Prompt buttons (Pass, End Turn, OK, etc.) -------------------------
    // Exclude UI-only take-back actions that would cause infinite loops.
    static $excludedModes = [10000 => true, 10001 => true, 10003 => true]; // Undo / Undo Block / Revert Turn
    $turnPhaseStr = $gs->turnPhase->turnPhase ?? '';
    foreach ($gs->playerPrompt->buttons ?? [] as $btn) {
        if (!isset($btn->mode)) continue;
        $mode  = intval($btn->mode);
        if (isset($excludedModes[$mode])) continue;
        $value = $btn->value ?? '';
        $moves[] = [
            'id'          => $nextID++,
            'type'        => 'BUTTON',
            'mode'        => $mode,
            'params'      => array_filter(['mode' => $mode, 'buttonInput' => $value], fn($v) => $v !== ''),
            'description' => $btn->text ?? "Button (mode $mode)",
        ];
    }

    // ---- INPUTCARDNAME decision --------------------------------------------
    // When the engine needs a card name (e.g. "name a card" effects), expose a
    // single move using mode 30. The card named doesn't matter strategically for
    // training — just pick the first card in hand so it's always deck-appropriate.
    if (($gs->turnPhase->turnPhase ?? '') === 'INPUTCARDNAME') {
        $handCards = $gs->playerHand ?? [];
        $namedCard = !empty($handCards) ? ($handCards[0]->cardNumber ?? 'Enlightened_Strike') : 'Enlightened_Strike';
        $moves[] = [
            'id'          => $nextID++,
            'type'        => 'INPUT_CARD_NAME',
            'mode'        => 30,
            'params'      => ['mode' => 30, 'buttonInput' => $namedCard],
            'description' => 'Name a card (' . $namedCard . ')',
        ];
    }

    // ---- Input popup -------------------------------------------------------
    $popup = $gs->playerInputPopUp ?? null;
    if ($popup && ($popup->active ?? false)) {
        foreach (PopupMoves($popup, $nextID) as $move) {
            $moves[] = $move;
        }
    }

    // ---- Pass phase --------------------------------------------------------
    if ($gs->canPassPhase ?? false) {
        // Only add if a pass button isn't already present from playerPrompt
        $hasPass = false;
        foreach ($moves as $m) {
            if (($m['mode'] ?? 0) === 99) { $hasPass = true; break; }
        }
        if (!$hasPass) {
            $moves[] = [
                'id'          => $nextID++,
                'type'        => 'PASS',
                'mode'        => 99,
                'params'      => ['mode' => 99],
                'description' => 'Pass current phase',
            ];
        }
    }

    // ---- Fallback: never return an empty move list --------------------------
    // Some DQ phases (e.g. CHOOSENUMBER, exotic triggers) leave no legal move
    // in the lists above.  Rather than handing the agent a zero-action state,
    // always provide at least one action so the episode can advance.
    if (empty($moves)) {
        $turnPhase = $gs->turnPhase->turnPhase ?? '';
        if ($turnPhase === 'P') {
            // Stuck in P phase with nothing to pitch (empty hand or all hand cards
            // filtered out by affordability).  Cancel undoes the play and returns
            // to M phase — the only escape from this deadlock.
            $moves[] = [
                'id'          => 0,
                'type'        => 'CANCEL',
                'mode'        => 10000,
                'params'      => ['mode' => 10000],
                'description' => 'Cancel (cannot pay cost)',
            ];
        } else {
            $moves[] = [
                'id'          => 0,
                'type'        => 'PASS',
                'mode'        => 99,
                'params'      => ['mode' => 99],
                'description' => 'Pass (fallback)',
            ];
        }
    }

    return $moves;
}

function CardMove(int $id, $card, string $zone): array
{
    $action   = intval($card->action ?? 0);
    $cardID   = $card->cardNumber ?? '';
    $override = (string)($card->actionDataOverride ?? $cardID);

    // params maps directly to ProcessInput.php GET params
    $params = ['mode' => $action, 'cardID' => $override];

    return [
        'id'          => $id,
        'type'        => ActionType($action),
        'mode'        => $action,
        'cardID'      => $cardID,
        'zone'        => $zone,
        'stats'       => CardStats($cardID),
        'params'      => $params,
        'description' => ActionDescription($action, $cardID, $zone),
    ];
}

/**
 * Extracts moves from an active playerInputPopUp.
 * Handles BUTTONINPUT, YESNO, and card-choice popups (CHOOSEMULTIZONE, etc.).
 */
function PopupMoves($popup, int &$nextID): array
{
    $moves = [];

    // Button choices in the popup (BUTTONINPUT, YESNO, modal choices)
    foreach ($popup->popup->buttons ?? [] as $btn) {
        if (!isset($btn->mode)) continue;
        $mode  = intval($btn->mode);
        $value = $btn->value ?? '';
        $moves[] = [
            'id'          => $nextID++,
            'type'        => 'POPUP_CHOICE',
            'mode'        => $mode,
            'params'      => array_filter(['mode' => $mode, 'buttonInput' => $value], fn($v) => $v !== ''),
            'description' => 'Choose: ' . ($btn->text ?? $value),
        ];
    }

    // Card choices in the popup (CHOOSEMULTIZONE etc.)
    // actionDataOverride on each card is the checkbox index to submit
    foreach ($popup->popup->cards ?? [] as $idx => $card) {
        if (($card->action ?? 0) === 0) continue;
        $override = $card->actionDataOverride ?? (string)$idx;
        $moves[] = [
            'id'          => $nextID++,
            'type'        => 'CHOOSE_CARD',
            'mode'        => 19,
            'cardID'      => $card->cardNumber ?? '',
            'stats'       => CardStats($card->cardNumber ?? ''),
            // Submit one card at a time; AI may submit multiple chk* for multi-select
            'params'      => ['mode' => 19, 'chkCount' => 1, 'chk0' => (string)$override],
            'description' => 'Choose card: ' . ($card->cardNumber ?? ''),
        ];
    }

    return $moves;
}

// ---------------------------------------------------------------------------
// State sections
// ---------------------------------------------------------------------------

function BuildMyState(stdClass $gs): array
{
    return [
        'health'     => $gs->playerHealth    ?? 0,
        'resources'  => $gs->playerPitchCount ?? 0,
        'ap'         => $gs->playerAP         ?? 0,
        'deckCount'  => $gs->playerDeckCount  ?? 0,
        'soulCount'  => $gs->playerSoulCount  ?? 0,
        'hand'       => CardList($gs->playerHand       ?? [], enriched: true),
        'arsenal'    => CardList($gs->playerArse        ?? [], enriched: true),
        'equipment'  => CardList($gs->playerEquipment  ?? [], enriched: true),
        'auras'      => CardList($gs->playerAuras       ?? [], enriched: true),
        'items'      => CardList($gs->playerItems       ?? [], enriched: true),
        'allies'     => CardList($gs->playerAllies      ?? [], enriched: true),
        'permanents' => CardList($gs->playerPermanents  ?? [], enriched: true),
        'discard'    => CardList($gs->playerDiscard     ?? []),
        'banish'     => CardList($gs->playerBanish      ?? []),
        'pitch'      => CardList($gs->playerPitch       ?? []),
    ];
}

function BuildTheirState(stdClass $gs): array
{
    return [
        'health'     => $gs->opponentHealth    ?? 0,
        'handCount'  => count((array)($gs->opponentHand ?? [])),
        'deckCount'  => $gs->opponentDeckCount ?? 0,
        'soulCount'  => $gs->opponentSoulCount ?? 0,
        'equipment'  => CardList($gs->opponentEquipment  ?? [], enriched: true),
        'auras'      => CardList($gs->opponentAuras       ?? [], enriched: true),
        'items'      => CardList($gs->opponentItems       ?? [], enriched: true),
        'allies'     => CardList($gs->opponentAllies      ?? [], enriched: true),
        'permanents' => CardList($gs->opponentPermanents  ?? [], enriched: true),
        'discard'    => CardList($gs->opponentDiscard     ?? []),
        'banish'     => CardList($gs->opponentBanish      ?? []),
        'pitch'      => CardList($gs->opponentPitch       ?? []),
    ];
}

function BuildCombatChain(stdClass $gs): ?array
{
    $cc = $gs->activeChainLink ?? null;
    if (!$cc || !isset($cc->attackingCard)) return null;

    return [
        'attackingCard'    => ($cc->attackingCard->cardNumber ?? null),
        'attackingCardStats' => CardStats($cc->attackingCard->cardNumber ?? ''),
        'totalPower'       => $cc->totalPower       ?? 0,
        'totalDefense'     => $cc->totalDefense     ?? 0,
        'goAgain'          => $cc->goAgain          ?? false,
        'dominate'         => $cc->dominate         ?? false,
        'overpower'        => $cc->overpower        ?? false,
        'piercing'         => $cc->piercing         ?? false,
        'phantasm'         => $cc->phantasm         ?? false,
        'wager'            => $cc->wager            ?? false,
        'damagePrevention' => $cc->damagePrevention ?? 0,
        'attackTarget'     => $cc->attackTarget     ?? [],
        'reactions'        => array_map(
            fn($c) => $c->cardNumber ?? '',
            $cc->reactions ?? []
        ),
    ];
}

function BuildStack(stdClass $gs): ?array
{
    $ld = $gs->layerDisplay ?? null;
    if (!$ld || empty($ld->layerContents ?? [])) return null;

    return [
        'target'   => $ld->target ?? [],
        'contents' => array_map(
            fn($c) => ['cardID' => $c->cardNumber ?? '', 'stats' => CardStats($c->cardNumber ?? '')],
            $ld->layerContents
        ),
    ];
}

function BuildPendingDecision(stdClass $gs): ?array
{
    $popup = $gs->playerInputPopUp ?? null;
    if (!$popup || !($popup->active ?? false)) return null;

    $phaseType = $gs->turnPhase->turnPhase ?? 'UNKNOWN';
    $options   = [];

    foreach ($popup->popup->buttons ?? [] as $btn) {
        $options[] = ['text' => $btn->text ?? '', 'value' => $btn->value ?? ''];
    }
    foreach ($popup->popup->cards ?? [] as $card) {
        $options[] = ['cardID' => $card->cardNumber ?? '', 'override' => $card->actionDataOverride ?? ''];
    }

    return [
        'type'    => $phaseType,
        'context' => $gs->turnPhase->caption ?? '',
        'options' => $options,
    ];
}

// ---------------------------------------------------------------------------
// Card helpers
// ---------------------------------------------------------------------------

/**
 * Converts an array of JSONRenderedCard objects to a simple AI-friendly list.
 * When $enriched is true, appends static card stats (cost, power, defense, pitch).
 */
function CardList(array $cards, bool $enriched = false): array
{
    return array_map(function ($c) use ($enriched) {
        $entry = [
            'cardID'        => $c->cardNumber  ?? '',
            'counters'      => $c->counters     ?? 0,
            'powerCounters' => $c->powerCounters ?? 0,
            'defCounters'   => $c->defCounters   ?? 0,
            'facing'        => $c->facing        ?? null,
            'tapped'        => $c->tapped        ?? false,
            'label'         => $c->label         ?? null,
        ];
        if ($enriched && !empty($entry['cardID'])) {
            $entry['stats'] = CardStats($entry['cardID']);
        }
        return $entry;
    }, $cards);
}

/**
 * Returns static card properties useful for AI evaluation.
 * Returns an empty array for blank/unknown card IDs.
 */
function CardStats(string $cardID): array
{
    if ($cardID === '' || $cardID === 'CARDBACK' || $cardID === 'BLANK') return [];

    return [
        'type'     => CardType($cardID),
        'subtype'  => CardSubtype($cardID),
        'cost'     => CardCost($cardID),
        'power'    => PowerValue($cardID),
        'defense'  => BlockValue($cardID),
        'pitch'    => PitchValue($cardID),
    ];
}

// ---------------------------------------------------------------------------
// Action type helpers
// ---------------------------------------------------------------------------

function ActionType(int $action): string
{
    return match ($action) {
        3  => 'ACTIVATE_EQUIPMENT',
        4  => 'ADD_TO_ARSENAL',
        5  => 'PLAY_FROM_ARSENAL',
        6  => 'PITCH',
        10 => 'ACTIVATE_ITEM',
        14 => 'PLAY_FROM_BANISH',
        15 => 'PLAY_FROM_OPP_BANISH',
        16 => 'CHOOSE_CARD',
        20 => 'YES_NO',
        21 => 'ACTIVATE_CHAIN_LINK',
        22 => 'ACTIVATE_AURA',
        23 => 'CHOOSE_CARD_OPT',
        24 => 'ACTIVATE_ALLY',
        25 => 'ACTIVATE_LANDMARK',
        27 => 'PLAY_FROM_HAND',
        28 => 'PITCH_FROM_HAND',
        34 => 'ACTIVATE_PERMANENT',
        35 => 'PLAY_FROM_DECK',
        36 => 'PLAY_FROM_GRAVEYARD',
        37 => 'PLAY_FROM_OPP_ARSENAL',
        38 => 'ACTIVATE_PAST_CHAIN_LINK',
        99 => 'OK',
        default => 'ACTION_' . $action,
    };
}

function ActionDescription(int $action, string $cardID, string $zone): string
{
    return match ($action) {
        3  => "Activate equipment: $cardID",
        4  => "Send to arsenal: $cardID",
        5  => "Play from arsenal: $cardID",
        6  => "Pitch: $cardID",
        10 => "Activate item: $cardID",
        14 => "Play from banish: $cardID",
        15 => "Play from opponent's banish: $cardID",
        16 => "Choose: $cardID",
        21 => "Activate from combat chain: $cardID",
        22 => "Activate aura: $cardID",
        24 => "Activate ally: $cardID",
        25 => "Activate landmark: $cardID",
        27 => "Play from hand: $cardID",
        28 => "Pitch from hand: $cardID",
        34 => "Activate permanent: $cardID",
        35 => "Play from top of deck: $cardID",
        36 => "Play from graveyard: $cardID",
        37 => "Play from opponent's arsenal: $cardID",
        38 => "Activate past chain link: $cardID",
        99 => 'OK',
        default => "Action $action on $cardID ($zone)",
    };
}