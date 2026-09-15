"""
Storage management for SarkariGPT
Safe JSON-based storage for user profiles, chat history, and bookmarks.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")
CHAT_HISTORY_PATH = os.path.join(DATA_DIR, "chat_history.json")
BOOKMARKS_PATH = os.path.join(DATA_DIR, "bookmarks.json")


def load_user_profile() -> Dict[str, Any]:
    """Load user profile from JSON file."""
    try:
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading user profile: {e}")
    return {}


def save_user_profile(profile_data: Dict[str, Any]) -> bool:
    """Save user profile to JSON file."""
    try:
        current = load_user_profile()
        current.update(profile_data)
        current["updated_at"] = datetime.now().isoformat()
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving user profile: {e}")
        return False


def reset_user_profile() -> bool:
    """Reset user profile."""
    try:
        if os.path.exists(PROFILE_PATH):
            os.remove(PROFILE_PATH)
        return True
    except Exception as e:
        print(f"Error resetting profile: {e}")
        return False


def load_chat_history() -> List[Dict[str, Any]]:
    """Load chat history from JSON file."""
    try:
        if os.path.exists(CHAT_HISTORY_PATH):
            with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chat history: {e}")
    return []


def save_chat_interaction(user_query: str, assistant_response: str, sources: Optional[List[str]] = None) -> bool:
    """Append a chat interaction to the persistent chat history."""
    try:
        history = load_chat_history()
        history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_query,
            "assistant": assistant_response,
            "sources": sources or []
        })
        # Keep the latest 100 exchanges
        history = history[-100:]
        with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving chat interaction: {e}")
        return False


def clear_chat_history() -> bool:
    """Clear chat history file."""
    try:
        if os.path.exists(CHAT_HISTORY_PATH):
            os.remove(CHAT_HISTORY_PATH)
        return True
    except Exception as e:
        print(f"Error clearing chat history: {e}")
        return False


def load_bookmarks() -> List[str]:
    """Load bookmarked scheme IDs."""
    try:
        if os.path.exists(BOOKMARKS_PATH):
            with open(BOOKMARKS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading bookmarks: {e}")
    return []


def toggle_bookmark(scheme_id: str) -> bool:
    """Toggle a scheme ID in bookmarks list."""
    try:
        bookmarks = load_bookmarks()
        if scheme_id in bookmarks:
            bookmarks.remove(scheme_id)
        else:
            bookmarks.append(scheme_id)
        with open(BOOKMARKS_PATH, "w", encoding="utf-8") as f:
            json.dump(bookmarks, f, indent=2)
        return True
    except Exception as e:
        print(f"Error toggling bookmark: {e}")
        return False
