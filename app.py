"""
SarkariGPT - AI Assistant for Indian Government Schemes
Full Semantic RAG application using ChromaDB, Sentence-Transformers, Groq Open-Source LLMs & Groq Whisper.
"""

import streamlit as st
import os
import json
import base64
from typing import Dict, List, Any
from rag_engine import SarkariRAGEngine
from ai_services import SarkariAIService, AVAILABLE_MODELS, LANGUAGE_NAMES, get_live_models
import storage

# Page Configuration
st.set_page_config(
    page_title="SarkariGPT - Indian Schemes AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .scheme-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .scheme-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
    }
    .badge-cat { background-color: #DBEAFE; color: #1E40AF; }
    .badge-level { background-color: #FEF3C7; color: #92400E; }
    .badge-match-high { background-color: #DCFCE7; color: #166534; }
    .badge-match-med { background-color: #FEF9C3; color: #854D0E; }
    .badge-match-low { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_profile" not in st.session_state:
    st.session_state.user_profile = storage.load_user_profile()
if "rag_engine" not in st.session_state:
    with st.spinner("Initializing ChromaDB Semantic Knowledge Base..."):
        st.session_state.rag_engine = SarkariRAGEngine()
if "selected_scheme_for_chat" not in st.session_state:
    st.session_state.selected_scheme_for_chat = None

# Sidebar Configuration
with st.sidebar:
    st.title("🏛️ SarkariGPT")
    st.caption("AI Assistant for Indian Government Schemes")
    st.divider()

    # Groq API Key Setup
    st.subheader("🔑 Groq API Setup")
    groq_api_key = ""
    try:
        groq_api_key = str(st.secrets["GROQ_API_KEY"]).strip()
        st.success("✅ Loaded API key from secrets")
    except Exception:
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.get("groq_api_key", ""),
            help="Get your free API key at console.groq.com/keys"
        )
        if groq_api_key:
            groq_api_key = groq_api_key.strip()
            st.session_state["groq_api_key"] = groq_api_key

    # Quick API Key inspection & validation test
    if groq_api_key:
        if not groq_api_key.startswith("gsk_"):
            st.warning("⚠️ Notice: Groq API keys usually begin with `gsk_`. Please check that you didn't paste a key from another service.")
        if st.button("🧪 Test API Key", use_container_width=True):
            test_service = SarkariAIService(api_key=groq_api_key)
            is_valid, msg = test_service.validate_api_key()
            if is_valid:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
    else:
        st.caption("👉 Get your free key at [console.groq.com/keys](https://console.groq.com/keys)")

    # Model Selector (Cloud-hosted Open Source LLMs)
    st.subheader("🤖 Open-Source Model")
    user_models = get_live_models(groq_api_key)
    selected_model_label = st.selectbox(
        "Select Model (Hosted on Groq Cloud)",
        options=list(user_models.keys()),
        index=0,
        help="Runs state-of-the-art open models on Groq's high-speed cloud LPU chips. Zero local hardware requirements."
    )
    chosen_model = user_models[selected_model_label]

    # Language Selector
    st.subheader("🌐 Language / भाषा")
    selected_lang_label = st.selectbox(
        "Response Language",
        options=list(LANGUAGE_NAMES.keys()),
        format_func=lambda x: LANGUAGE_NAMES[x],
        index=0
    )

    # Initialize AI Service
    ai_service = SarkariAIService(api_key=groq_api_key, model_name=chosen_model)

    st.divider()
    st.subheader("📚 Knowledge Base Stats")
    total_schemes = len(st.session_state.rag_engine.get_all_schemes())
    st.write(f"• Indexed Schemes: **{total_schemes} Flagship Schemes**")
    st.write("• Vector Store: **ChromaDB Persistent**")
    st.write("• Embeddings: **all-MiniLM-L6-v2 (384-dim)**")

    if st.button("🔄 Re-Index Knowledge Base", use_container_width=True):
        with st.spinner("Re-indexing schemes into ChromaDB..."):
            st.session_state.rag_engine.index_schemes(force_reload=True)
            st.success("Knowledge Base successfully re-indexed!")

    st.divider()
    st.caption("Developed with ❤️ for Indian Citizens • 100% Open Weights RAG")

# Main Page Header
st.markdown('<div class="main-title">🏛️ SarkariGPT - Indian Government Schemes Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Semantic RAG-powered guidance for central and state welfare schemes, grants, subsidies, and scholarships</div>', unsafe_allow_html=True)

# Tabs Navigation
tab_chat, tab_explorer, tab_comparator, tab_profile, tab_history = st.tabs([
    "💬 AI Assistant",
    "🔍 Scheme Explorer & Checker",
    "⚖️ Scheme Comparator",
    "👤 Citizen Profile",
    "📜 Chat History"
])

# -------------------------------------------------------------
# TAB 1: AI ASSISTANT (CHAT)
# -------------------------------------------------------------
with tab_chat:
    # Profile Banner
    profile = st.session_state.user_profile
    if profile and profile.get("name"):
        occ = profile.get("occupation", "Citizen")
        state = profile.get("state", "India")
        inc = f"₹{profile.get('annual_income', 0):,}" if profile.get("annual_income") is not None else "Not specified"
        st.info(f"👤 **Personalized for**: **{profile.get('name')}** | State: **{state}** | Occupation: **{occ}** | Income: **{inc}**")
    else:
        st.warning("💡 **Tip**: Fill in your **Citizen Profile** tab so SarkariGPT can accurately calculate your eligibility and give personalized scheme recommendations!")

    # Prompt Suggestions
    st.markdown("**Suggested Questions:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    suggested_q = None
    with col_p1:
        if st.button("🌾 Am I eligible for PM-KISAN?", use_container_width=True):
            suggested_q = "Am I eligible for PM-KISAN scheme and what are the application steps?"
    with col_p2:
        if st.button("🏥 How to get Ayushman Bharat card?", use_container_width=True):
            suggested_q = "How do I check eligibility and apply for the Ayushman Bharat PM-JAY health card?"
    with col_p3:
        if st.button("💼 Collateral-free business loans (MUDRA)", use_container_width=True):
            suggested_q = "What are the loan limits, interest rates, and documents required for PM Mudra Yojana?"

    # Check if a scheme was clicked from Explorer tab
    if st.session_state.selected_scheme_for_chat:
        suggested_q = f"Tell me full details about {st.session_state.selected_scheme_for_chat}. Am I eligible based on my profile, and how do I apply?"
        st.session_state.selected_scheme_for_chat = None

    # Render Conversation Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Knowledge Base Sources & Context"):
                    for s in msg["sources"]:
                        st.markdown(f"- [{s}]({s})")
            if msg.get("audio_data"):
                try:
                    audio_bytes = base64.b64decode(msg["audio_data"])
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception:
                    pass

    # Input Section: Text or Voice
    col_in1, col_in2 = st.columns([4, 1])
    with col_in2:
        voice_rec = st.audio_input("🎙️ Voice Question", key="voice_recorder")

    user_query = None
    with col_in1:
        text_query = st.chat_input("Ask any question about government schemes...")
        if text_query:
            user_query = text_query
        elif suggested_q:
            user_query = suggested_q

    # Handle Voice Input with Groq Whisper
    if voice_rec and not user_query:
        if not groq_api_key:
            st.error("Please enter your Groq API Key in the sidebar to enable Whisper voice transcription.")
        else:
            with st.spinner("🎙️ Transcribing voice using Groq Whisper Large v3..."):
                transcribed = ai_service.transcribe_audio(voice_rec.getvalue())
                if transcribed and not transcribed.startswith("Voice transcription error"):
                    st.success(f"Recognized: *'{transcribed}'*")
                    user_query = transcribed
                else:
                    st.error(transcribed)

    # Process Query
    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Assistant generation
        with st.chat_message("assistant"):
            if not groq_api_key:
                resp_text = "⚠️ **Groq API Key Required**: Please provide a free Groq API key in the left sidebar to enable AI answers."
                st.warning(resp_text)
                st.session_state.messages.append({"role": "assistant", "content": resp_text})
            else:
                with st.spinner("🔍 Retrieving verified scheme chunks from ChromaDB & Generating AI guidance..."):
                    # 1. Semantic Retrieval via ChromaDB
                    retrieved_chunks = st.session_state.rag_engine.semantic_search(
                        query=user_query,
                        n_results=5
                    )
                    
                    # 2. Open-source LLM generation via Groq
                    recent_history = storage.load_chat_history()
                    resp_text = ai_service.generate_rag_response(
                        user_query=user_query,
                        retrieved_chunks=retrieved_chunks,
                        user_profile=st.session_state.user_profile,
                        language_code=selected_lang_label,
                        chat_history=recent_history
                    )

                    st.markdown(resp_text)

                    # Extract unique sources
                    sources = []
                    for c in retrieved_chunks:
                        u = c.get("metadata", {}).get("url")
                        if u and u not in sources:
                            sources.append(u)

                    if sources:
                        with st.expander("📚 Verified Official Sources"):
                            for s in sources:
                                st.markdown(f"- [{s}]({s})")

                    # Voice synthesis for assistant if requested
                    audio_b64 = None
                    if voice_rec:
                        audio_data = ai_service.text_to_speech(resp_text, lang_code=selected_lang_label)
                        if audio_data:
                            audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                            st.audio(audio_data, format="audio/mp3")

                    # Save to session & persistent storage
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resp_text,
                        "sources": sources,
                        "audio_data": audio_b64
                    })
                    storage.save_chat_interaction(user_query, resp_text, sources)

# -------------------------------------------------------------
# TAB 2: SCHEME EXPLORER & ELIGIBILITY CHECKER
# -------------------------------------------------------------
with tab_explorer:
    st.header("🔍 Scheme Explorer & Personal Eligibility Checker")
    st.write("Browse all pre-seeded government schemes and see instant personal eligibility scores based on your citizen profile.")

    all_schemes = st.session_state.rag_engine.get_all_schemes()
    categories = ["All"] + sorted(list(set(s["category"] for s in all_schemes)))

    c_filter1, c_filter2 = st.columns([3, 1])
    with c_filter1:
        search_kw = st.text_input("Search schemes by keyword or benefit...", "")
    with c_filter2:
        cat_select = st.selectbox("Category Filter", categories)

    # Get ranked schemes with eligibility evaluation
    ranked_schemes = st.session_state.rag_engine.get_ranked_schemes_for_profile(
        profile=st.session_state.user_profile,
        query=search_kw
    )

    # Filter by category if selected
    if cat_select != "All":
        ranked_schemes = [item for item in ranked_schemes if item["scheme"]["category"] == cat_select]

    st.write(f"Showing **{len(ranked_schemes)}** schemes:")

    for item in ranked_schemes:
        scheme = item["scheme"]
        s_id = scheme["id"]
        elig_score = item["eligibility_score"]
        matches = item["matches"]
        warnings = item["warnings"]

        # Badge color based on eligibility
        if elig_score >= 80:
            badge_class = "badge-match-high"
            match_label = f"High Match: {elig_score}%"
        elif elig_score >= 50:
            badge_class = "badge-match-med"
            match_label = f"Medium Match: {elig_score}%"
        else:
            badge_class = "badge-match-low"
            match_label = f"Low Match: {elig_score}%"

        with st.container():
            st.markdown(f"""
            <div class="scheme-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; color: #1E3A8A;">{scheme['name']}</h3>
                    <span class="scheme-badge {badge_class}">{match_label}</span>
                </div>
                <div style="margin: 8px 0;">
                    <span class="scheme-badge badge-cat">{scheme['category']}</span>
                    <span class="scheme-badge badge-level">{scheme.get('level', 'Central')} Scheme</span>
                    <span style="font-size: 0.85rem; color: #6B7280;">Ministry: {scheme.get('ministry', 'N/A')}</span>
                </div>
                <p style="color: #374151; margin-bottom: 0.5rem;">{scheme.get('summary', '')}</p>
                <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">
                    <strong>💰 Key Benefit:</strong> {scheme.get('benefits', 'See scheme portal for details')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 6])
            with col_btn1:
                with st.popover("📋 View Eligibility & Docs"):
                    st.write(f"### {scheme['name']}")
                    st.write(f"**Eligibility Details:** {scheme.get('eligibility', {}).get('criteria', 'N/A')}")
                    
                    if matches:
                        st.success("✅ **Why you qualify:**\n" + "\n".join([f"- {m}" for m in matches]))
                    if warnings:
                        st.warning("⚠️ **Potential restrictions:**\n" + "\n".join([f"- {w}" for w in warnings]))

                    st.write("---")
                    st.write("**Required Documents:**")
                    for doc in scheme.get("documents_required", []):
                        st.write(f"- {doc}")
                    
                    st.write("---")
                    st.write(f"**Official Portal:** [{scheme.get('official_url')}]({scheme.get('official_url')})")

            with col_btn2:
                if st.button("💬 Ask in Chat", key=f"chat_with_{s_id}"):
                    st.session_state.selected_scheme_for_chat = scheme["name"]
                    st.toast(f"Selected {scheme['name']}! Opening Chat tab...")
                    st.rerun()

            st.write("")

# -------------------------------------------------------------
# TAB 3: SCHEME COMPARATOR
# -------------------------------------------------------------
with tab_comparator:
    st.header("⚖️ Side-by-Side Scheme Comparator")
    st.write("Compare two schemes to understand their financial assistance, income limits, and application processes side by side.")

    all_schemes = st.session_state.rag_engine.get_all_schemes()
    scheme_names = {s["name"]: s["id"] for s in all_schemes}

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        scheme1_name = st.selectbox("Select Scheme A", list(scheme_names.keys()), index=0)
    with comp_col2:
        scheme2_name = st.selectbox("Select Scheme B", list(scheme_names.keys()), index=min(1, len(scheme_names)-1))

    s1 = st.session_state.rag_engine.get_scheme_by_id(scheme_names[scheme1_name])
    s2 = st.session_state.rag_engine.get_scheme_by_id(scheme_names[scheme2_name])

    if s1 and s2:
        st.divider()
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.subheader(f"🅰️ {s1['name']}")
            st.write(f"**Category:** {s1['category']}")
            st.write(f"**Ministry:** {s1.get('ministry', 'N/A')}")
            st.info(f"**💰 Financial Benefits:**\n{s1.get('benefits', 'N/A')}")
            st.write(f"**Eligibility:** {s1.get('eligibility', {}).get('criteria', 'N/A')}")
            max_inc1 = s1.get('eligibility', {}).get('max_annual_income')
            st.write(f"**Income Ceiling:** {'₹' + f'{max_inc1:,}' + '/yr' if max_inc1 else 'No strict income ceiling'}")
            st.write("**Key Documents:**")
            for doc in s1.get("documents_required", [])[:4]:
                st.write(f"- {doc}")
            st.write(f"🔗 [Official Portal]({s1.get('official_url')})")

        with col_c2:
            st.subheader(f"🅱️ {s2['name']}")
            st.write(f"**Category:** {s2['category']}")
            st.write(f"**Ministry:** {s2.get('ministry', 'N/A')}")
            st.info(f"**💰 Financial Benefits:**\n{s2.get('benefits', 'N/A')}")
            st.write(f"**Eligibility:** {s2.get('eligibility', {}).get('criteria', 'N/A')}")
            max_inc2 = s2.get('eligibility', {}).get('max_annual_income')
            st.write(f"**Income Ceiling:** {'₹' + f'{max_inc2:,}' + '/yr' if max_inc2 else 'No strict income ceiling'}")
            st.write("**Key Documents:**")
            for doc in s2.get("documents_required", [])[:4]:
                st.write(f"- {doc}")
            st.write(f"🔗 [Official Portal]({s2.get('official_url')})")

# -------------------------------------------------------------
# TAB 4: CITIZEN PROFILE
# -------------------------------------------------------------
with tab_profile:
    st.header("👤 Citizen Profile & Eligibility Settings")
    st.write("Fill in your details below. Your information is stored locally on your device in secure JSON format.")

    curr_profile = st.session_state.user_profile

    indian_states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
        "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
        "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
        "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry", "Chandigarh"
    ]

    occupations = [
        "Farmer", "Student", "Daily Wage Worker", "Self Employed", "Business Owner",
        "Salaried Employee", "Private Employee", "Government Employee", "Unemployed",
        "Retired", "Professional"
    ]

    with st.form("profile_form"):
        p_col1, p_col2 = st.columns(2)

        with p_col1:
            name_input = st.text_input("Full Name", value=curr_profile.get("name", ""))
            state_idx = indian_states.index(curr_profile.get("state", "Maharashtra")) if curr_profile.get("state") in indian_states else 0
            state_input = st.selectbox("State of Residence", indian_states, index=state_idx)
            
            categories_list = ["General", "OBC", "SC", "ST", "EWS", "Minority"]
            cat_idx = categories_list.index(curr_profile.get("category", "General")) if curr_profile.get("category") in categories_list else 0
            cat_input = st.selectbox("Social Category", categories_list, index=cat_idx)
            
            gender_list = ["Male", "Female", "Other"]
            gen_idx = gender_list.index(curr_profile.get("gender", "Male")) if curr_profile.get("gender") in gender_list else 0
            gen_input = st.selectbox("Gender", gender_list, index=gen_idx)

        with p_col2:
            occ_idx = occupations.index(curr_profile.get("occupation", "Farmer")) if curr_profile.get("occupation") in occupations else 0
            occ_input = st.selectbox("Occupation", occupations, index=occ_idx)
            
            age_input = st.number_input("Age (Years)", min_value=1, max_value=120, value=int(curr_profile.get("age", 30)))
            income_input = st.number_input(
                "Annual Family Income (₹ INR)",
                min_value=0,
                max_value=50000000,
                value=int(curr_profile.get("annual_income", 150000)),
                step=10000,
                help="Total yearly household income used to check eligibility limits."
            )
            family_size = st.number_input("Family Size", min_value=1, max_value=25, value=int(curr_profile.get("family_size", 4)))

        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        if submitted:
            new_profile = {
                "name": name_input,
                "state": state_input,
                "category": cat_input,
                "gender": gen_input,
                "occupation": occ_input,
                "age": age_input,
                "annual_income": income_input,
                "family_size": family_size
            }
            storage.save_user_profile(new_profile)
            st.session_state.user_profile = new_profile
            st.success("✅ Profile successfully saved! Schemes eligibility has been updated.")
            st.rerun()

    if st.button("⚠️ Reset Profile"):
        storage.reset_user_profile()
        st.session_state.user_profile = {}
        st.success("Profile reset successfully.")
        st.rerun()

# -------------------------------------------------------------
# TAB 5: CHAT HISTORY
# -------------------------------------------------------------
with tab_history:
    st.header("📜 Saved Chat History")
    history_records = storage.load_chat_history()

    if history_records:
        st.write(f"Total interactions recorded: **{len(history_records)}**")
        for i, h in enumerate(reversed(history_records), 1):
            with st.expander(f"Q: {h.get('user')} ({h.get('timestamp', '')[:19]})"):
                st.markdown(h.get("assistant"))
                if h.get("sources"):
                    st.write("**Sources:**")
                    for s in h.get("sources"):
                        st.write(f"- [{s}]({s})")

        if st.button("🗑️ Clear History Log"):
            storage.clear_chat_history()
            st.session_state.messages = []
            st.success("Chat history cleared.")
            st.rerun()
    else:
        st.info("No recorded chat interactions yet.")
