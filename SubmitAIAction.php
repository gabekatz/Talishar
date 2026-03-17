<?php

/**
 * SubmitAIAction.php — AI action injection endpoint
 *
 * Accepts a JSON POST body, validates the action is legal for the current
 * game state, calls ProcessInput(), writes the updated gamestate, and
 * returns the next AI state.
 *
 * POST body (application/json):
 * {
 *   "gameName":    "12345",        // numeric game ID string
 *   "playerID":    1,              // 1 or 2
 *   "authKey":     "uuid",         // player's auth key
 *   "mode":        27,             // action mode (from legalMoves[n].mode)
 *   "cardID":      "0",            // zone index or card name (from legalMoves[n].params.cardID) — optional
 *   "buttonInput": "Option_A",     // button label (from legalMoves[n].params.buttonInput) — optional
 *   "chkCount":    0,              // number of checkbox selections — optional
 *   "chk":         ["0", "1"]      // checkbox indices (for CHOOSEMULTIZONE) — optional
 * }
 *
 * The `params` object from a GetAIState legalMove can be spread directly as
 * the POST body fields alongside gameName/playerID/authKey.
 *
 * Response:
 *   On success — same JSON shape as GetAIState (updated game state + next legal moves)
 *   On error   — { "error": "message" }
 *
 * Multi-select (CHOOSEMULTIZONE):
 *   When the phase requires choosing multiple cards, submit one action per card
 *   OR use chkCount + chk[] to submit all selections at once:
 *   { "mode": 19, "chkCount": 2, "chk": ["0", "2"] }
 */

ob_start();

error_reporting(E_ALL);

// CRITICAL: Capture session data and release the lock immediately.
if (session_status() === PHP_SESSION_NONE) session_start();
$sessionUserId = $_SESSION['userid'] ?? null;
session_write_close();

include 'WriteLog.php';
include 'GameLogic.php';
include 'GameTerms.php';
include 'HostFiles/Redirector.php';
include 'Libraries/SHMOPLibraries.php';
include 'Libraries/StatFunctions.php';
include 'Libraries/UILibraries.php';
include 'Libraries/PlayerSettings.php';
include 'Libraries/NetworkingLibraries.php';
include 'Libraries/CacheLibraries.php';
include 'AI/CombatDummy.php';
include 'Libraries/HTTPLibraries.php';
require_once 'Libraries/CoreLibraries.php';
include_once 'includes/dbh.inc.php';
include_once 'includes/functions.inc.php';
@include_once 'APIKeys/APIKeys.php';
include_once 'Libraries/ValidationLibraries.php';
include_once 'BuildGameState.php';
include_once 'BuildPlayerInputPopup.php';

SetHeaders();
header('Content-Type: application/json; charset=utf-8');

$body = json_decode(file_get_contents('php://input'), true) ?? [];

// ---- Input validation ------------------------------------------------------

$gameName = $body['gameName'] ?? '';
if (!IsGameNameValid($gameName)) {
    ob_clean();
    echo json_encode(['error' => 'Invalid game name.']);
    exit;
}

$playerID = intval($body['playerID'] ?? 0);
if ($playerID !== 1 && $playerID !== 2) {
    ob_clean();
    echo json_encode(['error' => 'playerID must be 1 or 2.']);
    exit;
}

$authKey = (string)($body['authKey'] ?? '');

$mode = intval($body['mode'] ?? 0);
if ($mode < 1 || $mode > 999999) {
    ob_clean();
    echo json_encode(['error' => 'mode must be an integer between 1 and 999999.']);
    exit;
}

$cardID     = isset($body['cardID'])     ? sanitizeString((string)$body['cardID'])     : '';
$buttonInput = isset($body['buttonInput']) ? sanitizeString((string)$body['buttonInput']) : '';
$chkCount   = intval($body['chkCount'] ?? 0);
$chkRaw     = $body['chk'] ?? [];

if ($chkCount < 0 || $chkCount > 100) {
    ob_clean();
    echo json_encode(['error' => 'chkCount must be between 0 and 100.']);
    exit;
}

$chkInput = [];
for ($i = 0; $i < $chkCount; ++$i) {
    $val = isset($chkRaw[$i]) ? sanitizeString((string)$chkRaw[$i]) : '';
    if ($val !== '') $chkInput[] = $val;
}

if ($cardID !== '' && !validateCardID($cardID)) {
    ob_clean();
    echo json_encode(['error' => 'Invalid cardID.']);
    exit;
}

if (!file_exists('./Games/' . $gameName . '/GameFile.txt')) {
    ob_clean();
    echo json_encode(['error' => 'Game not found.']);
    exit;
}

// ---- Load game state -------------------------------------------------------

include 'ParseGamestate.php';

$otherPlayer               = $currentPlayer == 1 ? 2 : 1;
$skipWriteGamestate        = false;
$mainPlayerGamestateStillBuilt = 0;
$makeCheckpoint            = 0;
$makeBlockBackup           = 0;
$MakeStartTurnBackup       = false;
$MakeStartGameBackup       = false;
$targetAuth                = ($playerID == 1 ? $p1Key : $p2Key);
$conceded                  = false;
$randomSeeded              = false;
$afterResolveEffects       = [];
$animations                = [];
$events                    = [];

// ---- Auth ------------------------------------------------------------------

if (!IsReplay()) {
    if ($authKey === '' && isset($_COOKIE['lastAuthKey'])) {
        $authKey = $_COOKIE['lastAuthKey'];
    }
    if ($authKey !== $targetAuth) {
        ob_clean();
        echo json_encode(['error' => 'Invalid auth key.']);
        exit;
    }
    if (!IsModeAsync($mode) && $currentPlayer != $playerID) {
        ob_clean();
        echo json_encode(['error' => 'Not your turn.']);
        exit;
    }
}

// ---- Legality check --------------------------------------------------------
// Verify that at least one legal move exists matching this mode.
// This is a lightweight guard; the engine's own ProcessInput is the authoritative checker.

if (!IsModeAsync($mode)) {
    $sessionData = [
        'userLoggedIn' => false, 'userName' => null,
        'isPvtVoidPatron' => false, 'patreonCampaigns' => [], 'friendList' => [],
    ];
    $gs       = BuildGameStateResponse($gameName, $playerID, $authKey, $sessionData, false);
    $isLegal  = false;
    if (!is_string($gs)) {
        $aiState = BuildAIState($gs, $playerID, $gameName);
        foreach ($aiState['legalMoves'] as $move) {
            if ($move['mode'] === $mode) {
                $isLegal = true;
                break;
            }
        }
    }
    if (!$isLegal) {
        ob_clean();
        echo json_encode(['error' => "Mode $mode is not a legal move in the current game state."]);
        exit;
    }
    // Re-parse gamestate because BuildGameStateResponse() may have side-effected globals
    ParseGamestate();
}

// ---- Mode 27: resolve hand index to card name (mirrors ProcessInput.php) ---

if ($mode === 27) {
    $hand       = GetHand($playerID);
    $index      = intval($cardID);
    $buttonInput = $hand[$index] ?? '';
}

// ---- Write replay command --------------------------------------------------

if (SaveReplay() && !IsReplay()) {
    $commandFile = fopen('./Games/' . $gameName . '/commandfile.txt', 'a');
    fwrite($commandFile, "$playerID $mode $buttonInput $cardID $chkCount " . implode('|', $chkInput) . "\r\n");
    fclose($commandFile);
}

// ---- Execute ---------------------------------------------------------------

ProcessInput($playerID, $mode, $buttonInput, $cardID, $chkCount, $chkInput, false, '');

ProcessMacros();

if ($inGameStatus == $GameStatus_Rematch) {
    // Rematch handling (mirrors ProcessInput.php)
    $origDeck = './Games/' . $gameName . '/p1DeckOrig.txt';
    if (file_exists($origDeck)) copy($origDeck, './Games/' . $gameName . '/p1Deck.txt');
    $origDeck = './Games/' . $gameName . '/p2DeckOrig.txt';
    if (file_exists($origDeck)) copy($origDeck, './Games/' . $gameName . '/p2Deck.txt');
    include 'MenuFiles/ParseGamefile.php';
    include 'MenuFiles/WriteGamefile.php';
    $gameStatus = (IsPlayerAI(2) ? $MGS_ReadyToStart : $MGS_ChooseFirstPlayer);
    SetCachePiece($gameName, 14, $gameStatus);
    $firstPlayer = 1;
    $firstPlayerChooser = ($winner == 1 ? 2 : 1);
    WriteLog("Player $firstPlayerChooser lost and will choose first player for the rematch.");
    WriteGameFile();
    $turn[0] = 'REMATCH';
    include 'WriteGamestate.php';
    $currentTime = round(microtime(true) * 1000);
    SetCachePiece($gameName, 2, $currentTime);
    SetCachePiece($gameName, 3, $currentTime);
    InvalidateGamestateCache($gameName);
    GamestateUpdated($gameName);
    ob_clean();
    echo json_encode(['message' => 'Rematch initiated.']);
    exit;
} elseif ($winner != 0 && $turn[0] != 'YESNO') {
    $inGameStatus    = $GameStatus_Over;
    $turn[0]         = 'OVER';
    $currentPlayer   = 1;
    global $events;
    $events = [];
}

CombatDummyAI();
if ($p2IsAI == '1') EncounterAI();
CacheCombatResult();

if (!IsGameOver()) {
    if ($playerID == 1) $p1TotalTime += time() - intval($lastUpdateTime);
    else                $p2TotalTime += time() - intval($lastUpdateTime);
    $lastUpdateTime = time();
}

// ---- Write gamestate -------------------------------------------------------

if (!$skipWriteGamestate) {
    if (!IsModeAsync($mode)) {
        $currentTime = round(microtime(true) * 1000);
        SetCachePiece($gameName, 12, '0');
        SetCachePiece($gameName, 2, $currentTime);
        SetCachePiece($gameName, 3, $currentTime);
        $currentPlayerActivity = 0;
    }
    DoGamestateUpdate();
    include 'WriteGamestate.php';
}

if ($makeCheckpoint)       MakeGamestateBackup();
if ($makeBlockBackup)      MakeGamestateBackup('preBlockBackup.txt');
if ($MakeStartTurnBackup)  MakeStartTurnBackup();
if ($MakeStartGameBackup)  MakeGamestateBackup('origGamestate.txt');

InvalidateGamestateCache($gameName);
GamestateUpdated($gameName);

// ---- Return updated AI state -----------------------------------------------

$sessionData = [
    'userLoggedIn' => false, 'userName' => null,
    'isPvtVoidPatron' => false, 'patreonCampaigns' => [], 'friendList' => [],
];

$updatedGs = BuildGameStateResponse($gameName, $playerID, $authKey, $sessionData, false);

ob_clean();

if (is_string($updatedGs)) {
    echo json_encode(['error' => $updatedGs]);
} else {
    echo json_encode(BuildAIState($updatedGs, $playerID, $gameName));
}

exit;

// ---------------------------------------------------------------------------
// All helpers below are identical to GetAIState.php.
// Kept here so SubmitAIAction.php is self-contained (no shared include needed).
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

function CollectLegalMoves(stdClass $gs): array
{
    $moves  = [];
    $nextID = 0;

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

    if (isset($gs->playerDeckCard) && ($gs->playerDeckCard->action ?? 0) !== 0) {
        $moves[] = CardMove($nextID++, $gs->playerDeckCard, 'DECK');
    }

    $currentResources = intval($gs->playerPitchCount ?? 0);
    $handSize = count((array)($gs->playerHand ?? []));
    $maxAffordable = $currentResources + $handSize * 3;

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
                if ($zone === 'HAND') {
                    $cardNum = $card->cardNumber ?? '';
                    $cost = max(0, (int)CardCost($cardNum));
                    if ($cost > 0) {
                        $cardPitch = max(0, (int)PitchValue($cardNum));
                        $affordableFromRemaining = $currentResources + ($totalHandPitch - $cardPitch);
                        if ($cost > $affordableFromRemaining) continue;
                    }
                }
                $moves[] = CardMove($nextID++, $card, $zone);
            }
        }
    }

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

    $chainLink = $gs->activeChainLink ?? null;
    if ($chainLink) {
        $chainCards = array_merge([$chainLink->attackingCard ?? null], $chainLink->reactions ?? []);
        foreach ($chainCards as $card) {
            if ($card && ($card->action ?? 0) !== 0) {
                $moves[] = CardMove($nextID++, $card, 'COMBAT_CHAIN');
            }
        }
    }

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

    $popup = $gs->playerInputPopUp ?? null;
    if ($popup && ($popup->active ?? false)) {
        foreach (PopupMoves($popup, $nextID) as $move) {
            $moves[] = $move;
        }
    }

    if ($gs->canPassPhase ?? false) {
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
    if (empty($moves)) {
        $turnPhase = $gs->turnPhase->turnPhase ?? '';
        if ($turnPhase === 'P') {
            // Stuck in P phase with nothing to pitch — Cancel undoes the play.
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
    $params   = ['mode' => $action, 'cardID' => $override];

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

function PopupMoves($popup, int &$nextID): array
{
    $moves = [];
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
    foreach ($popup->popup->cards ?? [] as $idx => $card) {
        if (($card->action ?? 0) === 0) continue;
        $override = $card->actionDataOverride ?? (string)$idx;
        $moves[] = [
            'id'          => $nextID++,
            'type'        => 'CHOOSE_CARD',
            'mode'        => 19,
            'cardID'      => $card->cardNumber ?? '',
            'stats'       => CardStats($card->cardNumber ?? ''),
            'params'      => ['mode' => 19, 'chkCount' => 1, 'chk0' => (string)$override],
            'description' => 'Choose card: ' . ($card->cardNumber ?? ''),
        ];
    }
    return $moves;
}

function BuildMyState(stdClass $gs): array
{
    return [
        'health'     => $gs->playerHealth     ?? 0,
        'resources'  => $gs->playerPitchCount ?? 0,
        'ap'         => $gs->playerAP          ?? 0,
        'deckCount'  => $gs->playerDeckCount   ?? 0,
        'soulCount'  => $gs->playerSoulCount   ?? 0,
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
        'attackingCard'      => ($cc->attackingCard->cardNumber ?? null),
        'attackingCardStats' => CardStats($cc->attackingCard->cardNumber ?? ''),
        'totalPower'         => $cc->totalPower       ?? 0,
        'totalDefense'       => $cc->totalDefense     ?? 0,
        'goAgain'            => $cc->goAgain          ?? false,
        'dominate'           => $cc->dominate         ?? false,
        'overpower'          => $cc->overpower        ?? false,
        'piercing'           => $cc->piercing         ?? false,
        'phantasm'           => $cc->phantasm         ?? false,
        'wager'              => $cc->wager            ?? false,
        'damagePrevention'   => $cc->damagePrevention ?? 0,
        'attackTarget'       => $cc->attackTarget     ?? [],
        'reactions'          => array_map(fn($c) => $c->cardNumber ?? '', $cc->reactions ?? []),
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

    $options = [];
    foreach ($popup->popup->buttons ?? [] as $btn) {
        $options[] = ['text' => $btn->text ?? '', 'value' => $btn->value ?? ''];
    }
    foreach ($popup->popup->cards ?? [] as $card) {
        $options[] = ['cardID' => $card->cardNumber ?? '', 'override' => $card->actionDataOverride ?? ''];
    }

    return [
        'type'    => $gs->turnPhase->turnPhase ?? 'UNKNOWN',
        'context' => $gs->turnPhase->caption ?? '',
        'options' => $options,
    ];
}

function CardList(array $cards, bool $enriched = false): array
{
    return array_map(function ($c) use ($enriched) {
        $entry = [
            'cardID'        => $c->cardNumber   ?? '',
            'counters'      => $c->counters      ?? 0,
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

function CardStats(string $cardID): array
{
    if ($cardID === '' || $cardID === 'CARDBACK' || $cardID === 'BLANK') return [];

    return [
        'type'    => CardType($cardID),
        'subtype' => CardSubtype($cardID),
        'cost'    => CardCost($cardID),
        'power'   => PowerValue($cardID),
        'defense' => BlockValue($cardID),
        'pitch'   => PitchValue($cardID),
    ];
}

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