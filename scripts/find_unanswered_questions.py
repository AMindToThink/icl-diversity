"""
Find Claude Code sessions where the last user-visible assistant message ends in a
question, with no real user reply afterward.

Usage: python find_unanswered.py [HOURS]   (default: 168 = 1 week)
"""
import json
import sys
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_DIR = Path("/home/cs29824/.claude/projects/-home-cs29824-matthew-icl-diversity")
HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 168

NOW = datetime.now(tz=timezone.utc)
CUTOFF = NOW - timedelta(hours=HOURS)

SIDECAR_TYPES = {
    "file-history-snapshot", "permission-mode", "system", "ai-title",
    "agent-name", "ai-input-summary", "custom-title", "last-prompt",
}

# Phrases the assistant uses when there was no real conversational turn
SIDECAR_ASSISTANT_TEXT = {
    "no response requested.",
    "",
}

def get_text(entry):
    if entry.get("type") not in ("user", "assistant"):
        return ""
    msg = entry.get("message", {})
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(cc.get("text", "") for cc in c if isinstance(cc, dict) and cc.get("type") == "text")
    return ""

def is_real_user_prompt(entry):
    if entry.get("type") != "user":
        return False
    msg = entry.get("message", {})
    c = msg.get("content")
    # Tool results aren't user prompts
    if isinstance(c, list) and all(isinstance(cc, dict) and cc.get("type") == "tool_result" for cc in c):
        return False
    text = get_text(entry).strip()
    if not text:
        return False
    # Filter out auto-generated user-side noise
    skip_prefixes = (
        "<bash-stdout>", "<bash-input>", "<system-reminder>",
        "<local-command-stdout>", "<local-command-caveat>",
        "[Request interrupted",
    )
    if any(text.startswith(p) for p in skip_prefixes):
        return False
    # Slash commands like /clear, /theme — count as real "user actions"
    return True

def get_ts(entry):
    ts = entry.get("timestamp")
    if not ts: return None
    try: return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except: return None

def find_last_real_assistant(entries):
    """Walk back to the last assistant message that was actually shown to the user."""
    for i in range(len(entries) - 1, -1, -1):
        e = entries[i]
        if e.get("type") != "assistant":
            continue
        text = get_text(e).strip()
        if text.lower() in SIDECAR_ASSISTANT_TEXT:
            continue
        if len(text) < 20:  # filter very short non-substantive replies
            continue
        return i, text
    return None, None

def is_question_ending(text):
    """Does the assistant's message end with a question that's plausibly awaiting an answer?"""
    # Strip trailing whitespace and common formatting
    t = text.rstrip()
    # Take last ~200 chars
    tail = t[-200:]
    # Find last sentence-ending punctuation
    # Look for ? in final sentence (last 200 chars)
    if "?" not in tail:
        return False, None
    # Find the actual final non-whitespace char of the message
    last_char = t[-1] if t else ""
    if last_char != "?":
        # Maybe it ends with a list item ending in ?, or a parenthesis after ?
        # Be lenient: ? within last 5 chars is fine
        if "?" not in t[-5:]:
            return False, None
    # Extract the last "sentence" (after last newline or sentence-end before the ?)
    # Simple: split on \n and . ! ? then return the chunk containing the trailing ?
    chunk = re.split(r'(?<=[.!?\n])\s+', t)[-1] if t else ""
    return True, chunk[-300:]

def main():
    print(f"Scanning sessions in the past {HOURS} hours ({CUTOFF.astimezone():%Y-%m-%d %H:%M})...\n")
    matches = []
    scanned = 0
    for path in PROJECT_DIR.glob("*.jsonl"):
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if mtime < CUTOFF:
            continue
        scanned += 1
        try:
            entries = []
            with open(path) as f:
                for raw in f:
                    try: entries.append(json.loads(raw))
                    except: pass
        except Exception as e:
            continue

        idx, asst_text = find_last_real_assistant(entries)
        if idx is None:
            continue
        asst_ts = get_ts(entries[idx])
        if asst_ts is None or asst_ts < CUTOFF:
            continue

        # Check no real user prompt came after this assistant message
        had_user_after = any(is_real_user_prompt(e) for e in entries[idx + 1:])
        if had_user_after:
            continue

        # Does it end with a question?
        is_q, last_chunk = is_question_ending(asst_text)
        if not is_q:
            continue

        cwd = next((e.get("cwd") for e in entries if e.get("cwd")), None)

        # Pull the most recent user prompt for context
        last_user_prompt = None
        for e in entries[:idx][::-1]:
            if is_real_user_prompt(e):
                last_user_prompt = get_text(e).strip()[:150]
                break

        matches.append({
            "session_id": path.stem,
            "ts": asst_ts,
            "cwd": cwd,
            "asst_tail": asst_text[-400:].strip(),
            "last_chunk": last_chunk.strip(),
            "last_user_prompt": last_user_prompt,
        })

    matches.sort(key=lambda m: m["ts"], reverse=True)
    print(f"Scanned {scanned} sessions. Found {len(matches)} ending with an unanswered question.\n")
    for m in matches:
        print("=" * 80)
        print(f"{m['ts'].astimezone():%Y-%m-%d %H:%M}  {m['session_id']}")
        print(f"cwd: {m['cwd']}")
        if m['last_user_prompt']:
            print(f"prior user: {m['last_user_prompt']!r}")
        print(f"final question: {m['last_chunk']}")
        print(f"resume: (cd {m['cwd']} && claude --resume {m['session_id']})")
    print("=" * 80)

if __name__ == "__main__":
    main()
