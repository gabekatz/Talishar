# Talishar Project Memory

## Project Overview
PHP backend game engine for Flesh and Blood (FaB) card game. Browser-based, file-backed game state.

## Key Architecture

### HTTP Endpoints (root level)
- `GetNextTurn.php` — poll game state (JSON), wraps `BuildGameStateResponse()`
- `ProcessInput.php` — GET-based action submission
- `ProcessInputAPI.php` — JSON POST, handles modes 26/33/106-109 (settings, layers, opt, triggers); default case is a no-op in the switch — normal card plays do NOT go here
- `GetAIState.php` — **NEW** AI state extraction endpoint (returns enriched state + legalMoves)
- `SubmitAIAction.php` — **NEW** JSON POST action injection for AI

### Game State Storage
- File: `./Games/{gameName}/GameFile.txt`
- Flat text, fixed line positions (see architecture docs)
- Parse: `ParseGamestate.php` / `ParseGamestate()`
- Write: `WriteGamestate.php`

### Legal Moves
- `IsPlayable()` in `CardDictionary.php:1708+` — authoritative playability check
- `BuildGameState.php` assigns `action` integers to each card (0=unplayable, non-zero=playable)
- Action integer = mode number to submit to ProcessInput.php
- `actionDataOverride` field = what to send as `cardID` GET param

### Action Submission Format (ProcessInput.php GET params)
```
gameName=X&playerID=Y&authKey=Z&mode=M&cardID=C&buttonInput=B&chkCount=N&chk0=V0...
```
Key modes:
- 27 = play from hand (cardID = hand index 0-based, engine converts to card name internally)
- 5  = play from arsenal (cardID = arsenal index)
- 6  = pitch from pitch zone
- 3  = activate equipment (cardID = character array index)
- 10/22/24/34 = activate item/aura/ally/permanent (cardID = zone index)
- 17 = BUTTONINPUT choice (buttonInput = option label, underscores as spaces)
- 19 = CHOOSEMULTIZONE (chkCount + chk0..N = checkbox indices)
- 20 = YES/NO (buttonInput = "YES" or "NO")
- 99 = pass phase

### Existing AI Framework (AI/ directory)
- `EncounterAI.php` — in-process AI, called after every ProcessInput for p2IsAI=="1"
- `CombatDummy.php` — for DUMMY hero character
- `EncounterPriorityValues.php` / `CardBehaviors.php` — priority system (0-0.9 per phase)
- `EncounterPlayLogic.php` — action submission helpers (PlayCardAttempt, BlockCardAttempt, etc.)
- `AIHelpers.php` — evaluation functions (EvaluateCardValue, ShouldBeAggressive, etc.)

## AI Integration Layer (Phases 1–3 complete)

### GetAIState.php (GET)
Params: `gameName`, `playerID`, `authKey`
Returns:
```json
{
  "gameID", "playerID", "turnNumber", "turnPlayer", "havePriority",
  "phase": { "turnPhase", "caption" },
  "myState": { "health", "resources", "ap", "deckCount", "hand[]", "arsenal[]", "equipment[]", "auras[]", "items[]", "allies[]", "permanents[]", "discard[]", "banish[]", "pitch[]" },
  "theirState": { "health", "handCount", ... },
  "combatChain": { "attackingCard", "totalPower", "totalDefense", "goAgain", "dominate", ... },
  "stack": { "target", "contents[]" },
  "pendingDecision": { "type", "context", "options[]" },
  "legalMoves": [ { "id", "type", "mode", "cardID?", "zone?", "stats?", "params", "description" } ]
}
```
Each `legalMove.params` can be spread directly as POST body to `SubmitAIAction.php`.
Cards in myState include `stats: { type, subtype, cost, power, defense, pitch }`.

### SubmitAIAction.php (JSON POST)
Body: `{ gameName, playerID, authKey, mode, cardID?, buttonInput?, chkCount?, chk? }`
- Validates auth and legality (mode must appear in current legalMoves)
- Calls ProcessInput(), ProcessMacros(), EncounterAI(), WriteGamestate
- Returns updated AI state (same shape as GetAIState)

## Card Object Patterns
- See `Classes/CardObjects/{SET}Cards.php` for card implementations
- PlayAbility, SpecificLogic, ProcessTrigger, CombatEffectActive, EffectPowerModifier
- Card IDs: lowercase_with_underscores
