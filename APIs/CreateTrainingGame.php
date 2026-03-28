<?php

/**
 * CreateTrainingGame.php — Programmatic game creation for AI training
 *
 * Combines CreateGame + Start into one call. Bypasses the lobby/session system.
 * Intended for local training only — no authentication required.
 *
 * POST body (application/json):
 * {
 *   "p1_deck":   "Ira",       // Deck file name (Assets/{name}.txt)
 *   "p2_deck":   "Ira",       // Deck file name for player 2
 *   "p2_is_ai":  true,        // true = use server-side EncounterAI for P2
 *   "format":    "cc"         // Game format string
 * }
 *
 * Response:
 * {
 *   "gameName":   "12345",
 *   "p1AuthKey":  "abc...",
 *   "p2AuthKey":  "def..."
 * }
 */

include_once "../HostFiles/Redirector.php";
include_once "../Libraries/HTTPLibraries.php";
include_once "../Libraries/SHMOPLibraries.php";
include_once "../Libraries/CacheLibraries.php";
include_once "../Libraries/NetworkingLibraries.php";
include_once "../Libraries/PlayerSettings.php";
include_once "../GameLogic.php";
include_once "../GameTerms.php";
include_once "../Libraries/StatFunctions.php";
include_once "../Libraries/UILibraries.php";
include_once "../AI/CombatDummy.php";
include_once "../MenuFiles/StartHelper.php";
include_once "../includes/dbh.inc.php";
include_once "../includes/functions.inc.php";

SetHeaders();
header('Content-Type: application/json; charset=utf-8');

ob_start();

/**
 * Fill a deck's main deck line from its inventory to reach the target size.
 *
 * Randomly selects cards from the inventory (lines 3+) and appends them to
 * the main deck (line 2) until it reaches $targetSize cards.  The selected
 * cards are removed from inventory.  This creates per-game deck variety
 * when used in training.
 *
 * @param string $deckFile   Path to the source deck .txt file
 * @param int    $targetSize Target main deck size (default 60 for CC)
 * @return string            Modified deck file content
 */
function fillDeckFromInventory(string $deckFile, int $targetSize = 60): string {
    $lines = file($deckFile, FILE_IGNORE_NEW_LINES);
    $heroLine  = $lines[0] ?? '';
    $deckCards = array_values(array_filter(explode(' ', trim($lines[1] ?? ''))));

    // Collect inventory cards (skip blank separator on line 3)
    $inventory = [];
    for ($i = 2; $i < count($lines); $i++) {
        $card = trim($lines[$i]);
        if ($card !== '') $inventory[] = $card;
    }

    // Fill main deck from inventory
    $needed = $targetSize - count($deckCards);
    if ($needed > 0 && count($inventory) > 0) {
        shuffle($inventory);
        $fill      = array_slice($inventory, 0, min($needed, count($inventory)));
        $deckCards  = array_merge($deckCards, $fill);
        $inventory  = array_slice($inventory, min($needed, count($inventory)));
    }

    // Shuffle the main deck so filled cards aren't always at the end
    shuffle($deckCards);

    // Rebuild deck file
    $result  = $heroLine . "\n";
    $result .= implode(' ', $deckCards) . "\n";
    $result .= "\n";
    foreach ($inventory as $card) {
        $result .= $card . "\n";
    }
    return $result;
}

$_POST = json_decode(file_get_contents('php://input'), true) ?? [];

$p1DeckName        = preg_replace('/[^a-zA-Z0-9_\-]/', '', $_POST['p1_deck'] ?? 'Ira');
$p2DeckName        = preg_replace('/[^a-zA-Z0-9_\-]/', '', $_POST['p2_deck'] ?? 'Ira');
$p2IsAI            = ($_POST['p2_is_ai'] ?? true)  ? '1' : '0';
$p1IsAI            = ($_POST['p1_is_ai'] ?? false) ? '1' : '0';
$format            = preg_replace('/[^a-z0-9]/', '', strtolower($_POST['format'] ?? 'cc'));
$fillFromInventory = !empty($_POST['fill_from_inventory']);
$targetDeckSize    = intval($_POST['target_deck_size'] ?? 60);

$p1DeckFile = "../Assets/{$p1DeckName}.txt";
$p2DeckFile = "../Assets/{$p2DeckName}.txt";

if (!file_exists($p1DeckFile)) {
    ob_clean();
    echo json_encode(['error' => "P1 deck file not found: Assets/{$p1DeckName}.txt"]);
    exit;
}
if (!file_exists($p2DeckFile)) {
    ob_clean();
    echo json_encode(['error' => "P2 deck file not found: Assets/{$p2DeckName}.txt"]);
    exit;
}

// ---- Create game directory -------------------------------------------------

$gameName = GetGameCounter("../");

if (!mkdir("../Games/$gameName", 0700, true)) {
    ob_clean();
    echo json_encode(['error' => 'Could not create game directory.']);
    exit;
}

// ---- Copy deck files (optionally filling main deck from inventory) ---------

if ($fillFromInventory) {
    file_put_contents("../Games/$gameName/p1Deck.txt", fillDeckFromInventory($p1DeckFile, $targetDeckSize));
    file_put_contents("../Games/$gameName/p2Deck.txt", fillDeckFromInventory($p2DeckFile, $targetDeckSize));
} else {
    copy($p1DeckFile, "../Games/$gameName/p1Deck.txt");
    copy($p2DeckFile, "../Games/$gameName/p2Deck.txt");
}

// ---- Generate auth keys and lobby metadata --------------------------------

$p1Key              = hash('sha256', rand() . rand());
$p2Key              = hash('sha256', rand() . rand() . rand());
$p1uid              = 'AI_P1';
$p2uid              = 'AI_P2';
$p1id               = '';
$p2id               = '';
$p1IsPatron         = '0';
$p2IsPatron         = '0';
$p1MetafyTiers      = [];
$p2MetafyTiers      = [];
$p1MetafyCommunities = [];
$p2MetafyCommunities = [];
$p1Data             = [1];
$p2Data             = [2];
$gameStatus         = 4; // Ready to start
$visibility         = 'private';
$firstPlayerChooser = '';
$firstPlayer        = 1;
$gameDescription    = 'Training Game';
$hostIP             = '127.0.0.1';
$joinerIP           = '127.0.0.1';
$p1DeckLink         = '';
$p2DeckLink         = '';
$p1IsChallengeActive = '0';
$p2IsChallengeActive = '0';
$p1deckbuilderID    = '';
$p2deckbuilderID    = '';
$roguelikeGameID    = '';
$p1StartingHealth   = '';
$p1ContentCreatorID = '';
$p2ContentCreatorID = '';
$p1SideboardSubmitted = '1';
$p2SideboardSubmitted = '1';
$p1StartingEquipment = [];
$p2StartingEquipment = [];
$p1Matchups         = [];
$p2Matchups         = [];
$gameGUID           = GenerateGameGUID();
$p1Inventory        = [];
$p2Inventory        = [];

// Write lobby GameFile.txt
$gameFileHandler = fopen("../Games/$gameName/GameFile.txt", 'w');
include "../MenuFiles/WriteGamefile.php";
WriteGameFile();

// Write empty gamelog
file_put_contents("../Games/$gameName/gamelog.txt", '');

// Initialize SHMOP cache (game visible in lobby)
$currentTime    = round(microtime(true) * 1000);
$cacheVisibility = '0'; // private
WriteCache($gameName,
    '1!' . $currentTime . '!' . $currentTime . '!0!-1!' . $currentTime .
    '!!!' . $cacheVisibility . '!0!0!0!' . $format . '!4!0!0'
);

// ---- Initialize gamestate.txt (mirrors Start.php) -------------------------

$filename = "../Games/$gameName/gamestate.txt";
$handler  = fopen($filename, 'w');
fwrite($handler, "20 20\r\n"); // Player life totals

$p1DeckHandler = fopen("../Games/$gameName/p1Deck.txt", 'r');
initializePlayerState($handler, $p1DeckHandler, 1);
fclose($p1DeckHandler);

$p2DeckHandler = fopen("../Games/$gameName/p2Deck.txt", 'r');
initializePlayerState($handler, $p2DeckHandler, 2);
fclose($p2DeckHandler);

fwrite($handler, "\r\n");         // Landmarks
fwrite($handler, "0\r\n");        // Winner
fwrite($handler, "$firstPlayer\r\n"); // First player
fwrite($handler, "1\r\n");        // Current player
fwrite($handler, "0\r\n");        // Current turn
fwrite($handler, "M 1\r\n");      // Phase / active player
fwrite($handler, "1\r\n");        // Action points
fwrite($handler, "\r\n");         // Combat chain
fwrite($handler, "0 0 0 0 0 0 0 GY NA 0 0 0 0 0 0 0 NA 0 0 -1 -1 NA 0 0 0 -1 0 0 0 0 - 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -1\r\n"); // Combat chain state
fwrite($handler, "\r\n");         // Current turn effects
fwrite($handler, "\r\n");         // Current turn effects from combat
fwrite($handler, "\r\n");         // Next turn effects
fwrite($handler, "\r\n");         // Decision queue
fwrite($handler, "0\r\n");        // Decision queue variables
fwrite($handler, "0 - - -\r\n"); // Decision queue state
fwrite($handler, "\r\n");         // Layers
fwrite($handler, "\r\n");         // Layer priority
fwrite($handler, "1\r\n");        // Main player
fwrite($handler, "\r\n");         // Last played card
fwrite($handler, "0\r\n");        // Number of prior chain links
fwrite($handler, "\r\n");         // Chain link summaries
fwrite($handler, "$p1Key\r\n");   // P1 auth key
fwrite($handler, "$p2Key\r\n");   // P2 auth key
fwrite($handler, "0\r\n");        // Permanent unique ID counter
fwrite($handler, "0\r\n");        // Game status (0=START)
fwrite($handler, "\r\n");         // Animations
fwrite($handler, "0\r\n");        // Current player activity
fwrite($handler, "0\r\n");        // P1 rating
fwrite($handler, "0\r\n");        // P2 rating
fwrite($handler, "0\r\n");        // P1 total time
fwrite($handler, "0\r\n");        // P2 total time
fwrite($handler, time() . "\r\n"); // Last update time
fwrite($handler, "$roguelikeGameID\r\n"); // Roguelike game id
fwrite($handler, "\r\n");         // Events
fwrite($handler, "-\r\n");        // Effect context
fwrite($handler, "\r\n");         // P1 inventory
fwrite($handler, "\r\n");         // P2 inventory
fwrite($handler, "$p1IsAI\r\n");  // Is P1 AI
fwrite($handler, "$p2IsAI\r\n");  // Is P2 AI
fclose($handler);

// Cache the initial gamestate
$gamestate = file_get_contents("../Games/$gameName/gamestate.txt");
WriteGamestateCache($gameName, $gamestate);

// ---- Run start effects -----------------------------------------------------

ob_clean(); // Suppress any include output

$MGS_GameStarted = 5;
chdir(dirname(__DIR__)); // ParseGamestate/WriteGamestate use relative ./Games/ paths
unset($filename);        // Let WriteGamestate.php rebuild with the new cwd
include "ParseGamestate.php";
include "StartEffects.php";

// Update lobby status to "started"
$gameStatus = $MGS_GameStarted;
$gameFileHandler = fopen("./Games/$gameName/GameFile.txt", 'r+');
WriteGameFile();

// Update cache to "game started"
$currentTime   = round(microtime(true) * 1000);
$currentUpdate = GetCachePiece($gameName, 1);
$cacheVisibility = '0';
WriteCache($gameName,
    ($currentUpdate + 1) . '!' . $currentTime . '!' . $currentTime .
    '!-1!-1!' . $currentTime . '!!!' . $cacheVisibility . '!0!0!0!' .
    $format . '!' . $MGS_GameStarted . '!0!0'
);

ob_clean();

echo json_encode([
    'gameName'   => (string)$gameName,
    'p1AuthKey'  => $p1Key,
    'p2AuthKey'  => $p2Key,
]);
exit;
