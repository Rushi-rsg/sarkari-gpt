"""
AI Services for SarkariGPT
Connects to Groq Cloud API for open-source model inference (Llama 3.3 70B, Llama 3.1 8B)
and Groq Whisper for multilingual Speech-to-Text.
"""

import io
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from gtts import gTTS

AVAILABLE_MODELS = {
    "Llama 3.3 70B (Recommended - High Accuracy)": "llama-3.3-70b-versatile",
    "Llama 3 70B": "llama3-70b-8192",
    "Llama 3 8B (Fast)": "llama3-8b-8192",
    "Mixtral 8x7B (MoE Architecture)": "mixtral-8x7b-32768",
    "Gemma 2 9B": "gemma2-9b-it"
}

def get_live_models(api_key: Optional[str] = None) -> Dict[str, str]:
    """Fetch active chat models available on the user's Groq account."""
    if not api_key:
        return AVAILABLE_MODELS
    try:
        temp_client = Groq(api_key=api_key.strip())
        models_resp = temp_client.models.list()
        live_models = {}
        for m in models_resp.data:
            m_id = m.id.lower()
            # Skip whisper audio and embedding models
            if any(term in m_id for term in ["whisper", "embed", "bge", "guard"]):
                continue

            raw_id = m.id
            if "llama-3.3-70b" in m_id:
                live_models["Llama 3.3 70B (Recommended - High Accuracy)"] = raw_id
            elif "llama-3.1-70b" in m_id:
                live_models["Llama 3.1 70B"] = raw_id
            elif "llama-3.1-8b" in m_id:
                live_models["Llama 3.1 8B (Fast)"] = raw_id
            elif "llama3-70b" in m_id:
                live_models["Llama 3 70B"] = raw_id
            elif "llama3-8b" in m_id:
                live_models["Llama 3 8B (Fast)"] = raw_id
            elif "mixtral" in m_id:
                live_models["Mixtral 8x7B (MoE)"] = raw_id
            elif "gemma" in m_id:
                live_models[f"Gemma ({raw_id})"] = raw_id
            elif "deepseek" in m_id:
                live_models[f"DeepSeek ({raw_id})"] = raw_id
            elif "qwen" in m_id:
                live_models[f"Qwen ({raw_id})"] = raw_id
            else:
                live_models[raw_id] = raw_id

        if live_models:
            # Prioritize Llama 3.3 70B at the top
            sorted_items = sorted(
                live_models.items(),
                key=lambda x: (0 if "3.3-70b" in x[1].lower() else 1, x[0])
            )
            return dict(sorted_items)
    except Exception as e:
        print(f"Failed to query live models: {e}")
    return AVAILABLE_MODELS

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "mr": "Marathi (मराठी)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "gu": "Gujarati (ગુજરાતી)"
}

class SarkariAIService:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key.strip() if api_key else None
        self.model_name = model_name
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip() if api_key else None
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def validate_api_key(self) -> Tuple[bool, str]:
        """Test API key with a minimal call."""
        if not self.client:
            return False, "API key is missing."
        try:
            self.client.models.list()
            return True, "API Key is valid and active!"
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "invalid_api_key" in err_str:
                return False, "Invalid API Key (401). Please create a fresh key at console.groq.com/keys."
            return False, f"API error: {err_str}"

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribe voice input using Groq's high-speed Whisper Large v3 model.
        Accurately handles Indian English and native Indian languages.
        """
        if not self.client:
            return "Error: Groq API key is missing. Please enter your API key in the sidebar."

        try:
            # Write bytes to a temporary wav/mp3 file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name

            with open(tmp_path, "rb") as file_obj:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(tmp_path), file_obj.read()),
                    model="whisper-large-v3",
                    temperature=0.0
                )
            
            os.unlink(tmp_path)
            return transcription.text.strip()
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "invalid_api_key" in err_str:
                return "Voice transcription error: Invalid Groq API Key (401). Please verify the key starts with 'gsk_' from console.groq.com/keys."
            return f"Voice transcription error: {err_str}"

    def text_to_speech(self, text: str, lang_code: str = "en") -> Optional[bytes]:
        """Convert text output to speech using gTTS."""
        try:
            # Clean markdown symbols for cleaner speech
            clean_text = text.replace("#", "").replace("*", "").replace("`", "").replace(">", "")
            # Limit TTS length to first 500 characters to prevent huge latency
            if len(clean_text) > 600:
                clean_text = clean_text[:600] + "... Please check the full response on your screen."
            
            tts_lang = lang_code if lang_code in ["en", "hi", "mr", "bn", "ta", "te", "gu"] else "en"
            tts = gTTS(text=clean_text, lang=tts_lang, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.getvalue()
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def generate_rag_response(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        language_code: str = "en",
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate grounded scheme recommendations using Groq-hosted open-source LLM.
        """
        if not self.client:
            return "⚠️ Please enter your Groq API Key in the left sidebar to enable AI responses."

        # Format retrieved context
        context_str = ""
        sources_list = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            meta = chunk.get("metadata", {})
            s_name = meta.get("scheme_name", "Unknown Scheme")
            url = meta.get("url", "")
            content = chunk.get("content", "")
            context_str += f"--- Source {i}: {s_name} ({url}) ---\n{content}\n\n"
            if url and url not in sources_list:
                sources_list.append(url)

        # Format user profile context
        profile_str = "Citizen Profile Information:\n"
        if user_profile:
            for k, v in user_profile.items():
                if v and k not in ["updated_at"]:
                    profile_str += f"- {k.replace('_', ' ').title()}: {v}\n"
        else:
            profile_str += "- No specific citizen profile entered yet.\n"

        # Format chat history
        history_str = ""
        if chat_history:
            recent = chat_history[-3:]
            for msg in recent:
                history_str += f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')[:300]}...\n"

        target_lang = LANGUAGE_NAMES.get(language_code, "English")

        system_prompt = f"""You are SarkariGPT, an expert and empathetic Indian Government Scheme Advisor.
Your objective is to provide accurate, up-to-date, and actionable guidance about welfare schemes, subsidies, pensions, and benefits.

CRITICAL INSTRUCTIONS:
1. Grounding: Rely strictly on the provided Context Information from official portals. If certain information is missing, state it honestly without hallucinating.
2. Eligibility Match: Check the Citizen Profile against the scheme eligibility rules (age, income ceiling, state, occupation, category). State clearly whether the citizen qualifies or not.
3. Response Language: Respond completely in {target_lang}. Use clear, polite, and accessible wording suitable for everyday Indian citizens.
4. Structure: Format your response using clean Markdown with these sections:
   - 🏛️ **Scheme Overview & Eligibility Verdict**: Summary and whether this user qualifies.
   - 💰 **Benefits & Financial Assistance**: Exact amounts, interest rates, or subsidies.
   - 📋 **Documents Required**: Bulleted checklist of necessary documents.
   - 🚀 **How to Apply**: Clear step-by-step instructions.
   - 🔗 **Official Portal**: Provide direct links and helpline info.
"""

        user_prompt = f"""{profile_str}

Recent Conversation:
{history_str if history_str else "None"}

Verified Context Information from Knowledge Base:
{context_str if context_str else "No direct matching scheme chunk found in database."}

Citizen Question:
"{user_query}"

Provide an authoritative, helpful response in {target_lang}.
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.2,
                max_tokens=1800
            )
            return chat_completion.choices[0].message.content or "No response generated."
        except Exception as e:
            err_str = str(e)
            if "model_not_found" in err_str or "does not exist" in err_str or "404" in err_str:
                fallback_model = "llama-3.3-70b-versatile" if self.model_name != "llama-3.3-70b-versatile" else "llama3-70b-8192"
                try:
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        model=fallback_model,
                        temperature=0.2,
                        max_tokens=1800
                    )
                    return f"> ℹ️ *Note: Model `{self.model_name}` was not found on your account; automatically answered using `{fallback_model}`.*\n\n" + (chat_completion.choices[0].message.content or "")
                except Exception:
                    pass
            if "401" in err_str or "invalid_api_key" in err_str:
                return "⚠️ **Invalid Groq API Key (401)**: The Groq API rejected this key. Please double-check your key in the left sidebar and ensure it begins with `gsk_` from [console.groq.com/keys](https://console.groq.com/keys)."
            return f"Error communicating with Groq API: {err_str}. Please select 'Llama 3.3 70B' in the sidebar."
