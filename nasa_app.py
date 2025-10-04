import os
import base64
import requests
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import PyPDF2
import docx
import streamlit.components.v1 as components 

if 'show_similarity_graph' not in st.session_state:
    st.session_state.show_similarity_graph = False
if 'show_knowledge_graph' not in st.session_state:
    st.session_state.show_knowledge_graph = False

# =========================
# 🎬 NASA MEDIA API FUNCTIONS - IMPROVED
# =========================
def search_nasa_media(query, max_results=3, media_type="image"):
    """
    Cari media (image/video) dari NASA API dengan fallback strategy
    Priority: images > videos (images lebih reliable)
    """
    url = f"https://images-api.nasa.gov/search?q={query}&media_type={media_type}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get("collection", {}).get("items", [])
            media_list = []
            
            for item in items[:max_results]:
                try:
                    nasa_id = item["data"][0].get("nasa_id", "")
                    title = item["data"][0].get("title", "No Title")
                    description = item["data"][0].get("description", "")[:150]
                    
                    # Get thumbnail/image URL
                    media_url = None
                    if "links" in item and item["links"]:
                        media_url = item["links"][0].get("href", "")
                    
                    # Get collection link
                    collection_url = f"https://images.nasa.gov/details/{nasa_id}"
                    
                    if media_url:  # Only add if we have valid media
                        media_list.append({
                            "nasa_id": nasa_id,
                            "title": title,
                            "description": description,
                            "media_url": media_url,
                            "collection_url": collection_url,
                            "media_type": media_type
                        })
                except Exception as e:
                    continue
                    
            return media_list
    except Exception as e:
        st.warning(f"NASA API connection issue: {e}")
    return []

def extract_keywords_from_text(text, max_keywords=3):
    """Extract meaningful keywords dari article text untuk search yang lebih baik"""
    if not text or len(text) < 50:
        return []
    
    # Common NASA/space science terms to prioritize
    priority_terms = [
        'microgravity', 'radiation', 'ISS', 'space station', 'astronaut',
        'Mars', 'Moon', 'lunar', 'solar', 'cosmic', 'plant', 'cell',
        'biology', 'protein', 'DNA', 'gene', 'muscle', 'bone', 'calcium'
    ]
    
    # Simple keyword extraction
    words = text.lower().split()
    keywords = []
    
    # Check for priority terms first
    for term in priority_terms:
        if term.lower() in text.lower() and term not in keywords:
            keywords.append(term)
            if len(keywords) >= max_keywords:
                break
    
    # If not enough keywords, get from title/first sentences
    if len(keywords) < max_keywords:
        important_words = [w for w in words if len(w) > 6 and w.isalpha()][:5]
        for word in important_words:
            if word not in keywords and len(keywords) < max_keywords:
                keywords.append(word)
    
    return keywords if keywords else ['space', 'research']

def display_nasa_media_gallery(media_list):
    """
    Display NASA media (images/videos) dalam gallery format yang cantik
    """
    if not media_list:
        st.info("🔍 No related NASA media found for this topic")
        return
    
    # Display media dalam columns
    cols = st.columns(min(len(media_list), 3))
    
    for idx, media in enumerate(media_list):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(0, 100, 200, 0.3) 0%, rgba(0, 50, 150, 0.3) 100%);
                padding: 12px;
                border-radius: 10px;
                border: 1px solid #00d4ff;
                margin: 8px 0;
                transition: transform 0.2s;
            ">
                <div style="color: #00f5ff; font-weight: bold; font-size: 0.9em; margin-bottom: 8px;">
                    {'🖼️' if media['media_type'] == 'image' else '🎬'} {media['title'][:40]}{'...' if len(media['title']) > 40 else ''}
                </div>
                <div style="color: #e0f7fa; font-size: 0.8em; margin-bottom: 10px;">
                    {media['description'][:80]}{'...' if len(media['description']) > 80 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display actual image/video
            if media['media_url']:
                try:
                    if media['media_type'] == 'image':
                        st.image(media['media_url'], use_container_width=True)
                    else:
                        st.video(media['media_url'])
                except:
                    st.markdown(f"[🔗 View on NASA.gov]({media['collection_url']})")
            
            # Link to full collection
            st.markdown(f"""
            <a href="{media['collection_url']}" target="_blank" style="
                color: #00f5ff;
                text-decoration: none;
                font-size: 0.8em;
                display: block;
                text-align: center;
                margin-top: 8px;
            ">🚀 View Full Collection →</a>
            """, unsafe_allow_html=True)
# =========================
# 🎨 Background & Sidebar Setup (SAMA SEPERTI ASAL)
# =========================
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c0e2e 0%, #1a237e 50%, #283593 100%);
    }
    
    .sidebar-content {
        color: #ffffff;
    }
    
    .stButton > button {
        color: #00ffff;
        border: 1px solid #00ffff;
        background: rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        background: rgba(0, 255, 255, 0.2);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_file = "background.jpg"
if os.path.exists(img_file):
    img_base64 = get_base64_of_bin_file(img_file)
    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0, 0, 0, 0);
    }}
    .block-container {{
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 200, 255, 0.5);
        box-shadow: 0 0 15px rgba(0, 200, 255, 0.25);
        color: #f0faff;
        transition: all 0.3s ease-in-out;
    }}
    .block-container:hover {{
        border: 1px solid rgba(0, 255, 255, 0.9);
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.6);
        transform: translateY(-3px);
    }}
    h1, h2, h3, h4 {{
        color: #1a3c34;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)
# =========================
# 🎵 Background Music - Auto-play setelah click page
# =========================
music_file = "background.mp3"
if os.path.exists(music_file):
    with open(music_file, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()

    md_audio = f"""
    <audio id="bgMusic" loop style="display: none">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    <script>
        let musicPlayed = false;
        
        document.addEventListener('click', function() {{
            if (!musicPlayed) {{
                const audio = document.getElementById('bgMusic');
                audio.volume = 0.3;
                audio.play()
                    .then(() => {{
                        musicPlayed = true;
                        console.log('Music started successfully');
                    }})
                    .catch(error => {{
                        console.log('Music play failed:', error);
                        // Create fallback button
                        createMusicButton();
                    }});
            }}
        }});
        
        function createMusicButton() {{
            if (!document.getElementById('musicBtn')) {{
                const btn = document.createElement('button');
                btn.id = 'musicBtn';
                btn.innerHTML = '🎵 Click to Play Music';
                btn.style.position = 'fixed';
                btn.style.bottom = '10px';
                btn.style.right = '10px';
                btn.style.zIndex = '9999';
                btn.style.padding = '10px';
                btn.style.background = '#00f5ff';
                btn.style.color = 'black';
                btn.style.border = 'none';
                btn.style.borderRadius = '5px';
                btn.style.cursor = 'pointer';
                
                btn.onclick = function() {{
                    const audio = document.getElementById('bgMusic');
                    audio.play();
                    document.body.removeChild(btn);
                }};
                
                document.body.appendChild(btn);
            }}
        }}
        
        // Try auto-play after 2 seconds
        setTimeout(() => {{
            const audio = document.getElementById('bgMusic');
            audio.play().catch(e => console.log('Auto-play failed, waiting for user click'));
        }}, 2000);
    </script>
    """
    components.html(md_audio, height=0)
# =========================
# 📊 Data Loading & AI Functions (SAMA SEPERTI ASAL)
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("SB_publication_PMC.csv", encoding="latin1")
    df.columns = [c.lower() for c in df.columns]
    return df

df = load_data()

def ai_summarize(text, max_tokens=400, mode="single"):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.sidebar.warning("⚠️ OpenRouter API key not found. Using rule-based summary.")
        return fallback_summarize(text, mode)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if mode == "overview":
        truncated_text = text[:6000]
        user_prompt = f"""
Berdasarkan koleksi abstrak dan kesimpulan dari artikel-artikel NASA tentang space bioscience, 
tuliskan ringkasan komprehensif dalam Bahasa Inggeris (5-10 paragraph). 
Fokus pada tema utama, penemuan penting, knowledge gaps, dan trend penelitian terkini.

Text:
{truncated_text}
"""
    else:
        truncated_text = text[:3000]
        user_prompt = f"""
Ringkaskan artikel NASA ini dalam 5-7 ayat dalam Bahasa Inggeris. 
Fokus pada objektif, metodologi, penemuan utama, dan implikasi:

{truncated_text}
"""

    data = {
        "model": "openrouter/auto",
        "messages": [
            {"role": "system", "content": "You are a scientific research assistant that summarizes NASA space bioscience articles clearly and concisely in English."},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return fallback_summarize(text, mode)
            
    except Exception as e:
        return fallback_summarize(text, mode)

def fallback_summarize(text, mode="single"):
    text = ' '.join(text.split())
    sentences = [s.strip() for s in text.split('. ') if len(s.strip()) > 10]
    
    if len(sentences) <= 3:
        return text
    
    if mode == "overview":
        if len(sentences) >= 10:
            key_indices = [0, 1, len(sentences)//3, len(sentences)//2, 
                          len(sentences)*2//3, -2, -1]
        else:
            key_indices = [0, len(sentences)//2, -1]
        
        key_sentences = [sentences[i] for i in key_indices if i < len(sentences)]
        summary = ". ".join(key_sentences) + "."
        return f"📋 **Rule-based Overview Summary:**\n\n{summary}"
    else:
        if len(sentences) <= 5:
            summary = ". ".join(sentences) + "."
        else:
            science_terms = ['study', 'research', 'results', 'found', 'conclusion', 
                           'method', 'data', 'analysis', 'significant', 'effect']
            
            scored_sentences = []
            for i, sentence in enumerate(sentences):
                score = 0
                if i < 2:
                    score += 2
                if i > len(sentences) - 3:
                    score += 2
                if any(term in sentence.lower() for term in science_terms):
                    score += 1
                if 50 < len(sentence) < 200:
                    score += 1
                    
                scored_sentences.append((score, sentence))
            
            scored_sentences.sort(reverse=True)
            top_sentences = [s[1] for s in scored_sentences[:4]]
            summary = ". ".join(top_sentences) + "."
        
        return f"📄 **Rule-based Article Summary:**\n\n{summary}"

def smart_summarize(text, max_tokens=400, mode="single"):
    use_ai = st.sidebar.checkbox("🤖 Use AI Summarization", value=True, 
                                help="Uncheck to use faster rule-based summaries")
    
    if use_ai:
        with st.spinner("🔄 Generating AI summary..."):
            ai_result = ai_summarize(text, max_tokens, mode)
        
        if any(keyword in ai_result.lower() for keyword in ['error', 'unavailable', 'timeout', 'network', 'fallback']):
            st.sidebar.warning("AI summarization failed. Using rule-based fallback.")
            return fallback_summarize(text, mode)
        
        return ai_result
    else:
        return fallback_summarize(text, mode)

# PDF/DOCX Functions (SAMA)
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def ai_comment_on_report(report_text, corpus_texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([report_text] + corpus_texts)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    most_similar_idx = sims.argmax()
    score = sims[most_similar_idx]
    related_paper = df.iloc[most_similar_idx]["title"]
    return f"Closest related article: **{related_paper}** (similarity score {score:.2f})."

# Graph Functions (SAMA)
def build_knowledge_graph(df, max_nodes=50):
    G = nx.Graph()
    for idx, row in df.head(max_nodes).iterrows():
        article = f"📄 {row['title'][:40]}..."
        G.add_node(article, color="lightblue", size=20)
        if "abstract" in row and pd.notna(row["abstract"]):
            words = str(row["abstract"]).split()
            keywords = [w for w in words if len(w) > 6][:5]
            for kw in keywords:
                G.add_node(kw, color="lightgreen", size=15)
                G.add_edge(article, kw)
    return G

def build_similarity_graph(df, max_nodes=30, top_k=3):
    G = nx.Graph()
    subset = df.head(max_nodes).copy()
    subset["text"] = subset["abstract"].fillna("") + " " + subset["conclusion"].fillna("")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(subset["text"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    for i, row in subset.iterrows():
        art_i = f"📄 {row['title'][:40]}..."
        G.add_node(art_i, color="orange", size=20)
        similar_idx = sim_matrix[i - subset.index[0]].argsort()[-top_k-1:-1]
        for j in similar_idx:
            art_j = f"📄 {subset.iloc[j]['title'][:40]}..."
            sim_score = sim_matrix[i - subset.index[0], j]
            if sim_score > 0.1:
                G.add_edge(art_i, art_j, weight=sim_score)
    return G

def display_knowledge_graph_online(df):
    try:
        G = build_knowledge_graph(df, max_nodes=30)
        net = Network(height="600px", width="100%", bgcolor="#0d1b2a", font_color="white")
        net.from_nx(G)
        html_content = net.generate_html()
        components.html(html_content, height=600, scrolling=True)
        return True
    except Exception as e:
        st.error(f"❌ Error generating graph: {e}")
        return False

def display_similarity_graph_online(df):
    try:
        G = build_similarity_graph(df, max_nodes=20)
        net = Network(height="600px", width="100%", bgcolor="#0d1b2a", font_color="white")
        net.from_nx(G)
        html_content = net.generate_html()
        components.html(html_content, height=600, scrolling=True)
        return True
    except Exception as e:
        st.error(f"❌ Error generating similarity graph: {e}")
        return False

# =========================
# 🖥️ Main UI (DENGAN VIDEO YANG DIUBAHSUAI)
# =========================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, rgba(13, 59, 102, 0.8) 0%, rgba(30, 110, 167, 0.8) 50%, rgba(42, 157, 143, 0.8) 100%);
        padding: 30px;
        border-radius: 20px;
        border: 2px solid #00f5ff;
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.4);
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 15px;
        border: 1px solid rgba(0, 245, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px 10px 0 0;
        padding: 15px 25px;
        border: 1px solid rgba(0, 245, 255, 0.2);
        color: #e0f7fa;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00f5ff 0%, #0077b6 100%);
        color: #001219;
        border: 1px solid #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.6);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 245, 255, 0.2);
        color: #00f5ff;
        border: 1px solid #00f5ff;
    }
    
    .main-content {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header (SAMA)
logo_file = "team_logo.png"
if os.path.exists(logo_file):
    with open(logo_file, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    logo_img = f'<img src="data:image/png;base64,{logo_base64}" class="team-logo" alt="Team Logo" style="width: 80px; height: 80px; border-radius: 50%; border: 2px solid #00f5ff;">'
else:
    logo_img = '<div style="width: 80px; height: 80px; border-radius: 50%; border: 2px solid #00f5ff; background: rgba(0,245,255,0.2); display: flex; align-items: center; justify-content: center; color: #00f5ff; font-weight: bold;">TEAM</div>'

st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-bottom: 15px;">
        {logo_img}
        <div>
            <h1 style="color: #00f5ff; text-shadow: 0 0 20px #00f5ff; margin-bottom: 10px; font-size: 2.5em;">
                🌌 NASA SPACE BIOSCIENCE DASHBOARD
            </h1>
            <p style="color: #e0f7fa; font-size: 1.3em; margin-bottom: 15px;">
                Advanced Analytics for Space Research Publications
            </p>
        </div>
        {logo_img}
    </div>
    <div style="display: flex; justify-content: center; gap: 20px; color: #e0f7fa;">
        <span>📊 <strong>608</strong> Publications</span>
        <span>🔬 <strong>12+</strong> Research Fields</span>
        <span>🤖 <strong>AI-Powered</strong> Analysis</span>
        <span>🚀 <strong>NASA</strong> Curated</span>
    </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔍 **SEARCH & ANALYZE**", "📑 **UPLOAD & COMPARE**"])

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# --- TAB 1: Search Publications (DENGAN VIDEO INTEGRATION) ---
with tabs[0]:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(13, 59, 102, 0.8) 0%, rgba(30, 110, 167, 0.6) 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
        margin-bottom: 25px;
    ">
        <h2 style="color: #00f5ff; margin: 0; text-align: center;">🔍 Advanced Publication Search</h2>
        <p style="color: #e0f7fa; text-align: center; margin: 10px 0 0 0;">
            Explore 608+ NASA Space Bioscience Publications with AI-Powered Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_stats, col_search = st.columns([1, 2])

    with col_stats:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(0, 245, 255, 0.3);
            height: 100%;
        ">
            <h4 style="color: #00f5ff; margin-top: 0;">📈 Publication Analytics</h4>
            <div style="color: #e0f7fa; line-height: 2;">
                📚 <strong>Total Articles:</strong> 608<br>
                📅 <strong>Years Covered:</strong> 1990-2024<br>
                🔬 <strong>Research Fields:</strong> 12+<br>
                🌟 <strong>Featured Topics:</strong><br>
                &nbsp;&nbsp;• Microgravity Effects<br>
                &nbsp;&nbsp;• Space Radiation<br>
                &nbsp;&nbsp;• Life Support Systems<br>
                &nbsp;&nbsp;• Astronaut Health<br>
                🤖 <strong>AI Tools:</strong> Available
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_search:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.08);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(0, 245, 255, 0.2);
            margin-bottom: 20px;
        ">
            <h4 style="color: #00f5ff; margin-top: 0;">🎯 Search Controls</h4>
        </div>
        """, unsafe_allow_html=True)
        
        years = sorted(df["year"].dropna().unique())
        selected_years = st.multiselect(
            "**🗓️ Filter by Publication Years:**",
            years, 
            default=years[-5:],
            help="Select specific years to focus your search"
        )
        
        query = st.text_input(
            "**🔍 Search Keywords:**",
            placeholder="Enter title, abstract, or conclusion keywords...",
            help="Search across article titles, abstracts, and conclusions"
        )
        
        with st.expander("💡 **Search Tips**", expanded=False):
            st.markdown("""
            - Use **specific terms**: "microgravity effects on cells"
            - Try **NASA acronyms**: "ISS, EVA, LEO"
            - Search **research fields**: "radiation biology", "plant growth"
            - Use **boolean operators**: "Mars AND habitat"
            """)

    # Results Section dengan NASA Video Integration
    if query:
        df_filtered = df[df["year"].isin(selected_years)] if selected_years else df
        results = df_filtered[df_filtered.astype(str)
                              .apply(lambda x: x.str.contains(query, case=False, na=False))
                              .any(axis=1)]
        
        st.markdown(f"""
        <div style="
            background: rgba(0, 245, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 4px solid #00f5ff;
        ">
            <h4 style="color: #00f5ff; margin: 0;">
                📊 Search Results: <span style="color: #e0f7fa;">{len(results)} articles found</span>
            </h4>
            <p style="color: #e0f7fa; margin: 5px 0 0 0;">
                Keywords: "{query}" | Years: {len(selected_years)} selected
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, row in results.head(20).iterrows():
            colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", "#FF9FF3", "#54A0FF", "#5F27CD"]
            current_color = colors[idx % len(colors)]
            
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.03);
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                border-left: 5px solid {current_color};
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease;
            ">
                <h3 style="color: {current_color}; margin: 0 0 10px 0;">📄 {row['title']}</h3>
                <div style="color: #e0f7fa; font-size: 0.9em; margin-bottom: 15px;">
                    🗓️ <strong>Year:</strong> {row['year'] if 'year' in row and pd.notna(row['year']) else 'N/A'} 
                    | 🔗 <strong>Access:</strong> {'Available' if 'link' in row and pd.notna(row['link']) else 'Not available'}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if "link" in row and pd.notna(row["link"]):
                st.markdown(f"**🌐 Full Article:** [{row['link']}]({row['link']})")

            if "abstract" in row and pd.notna(row["abstract"]):
                with st.expander("📖 **Abstract**", expanded=False):
                    st.write(row["abstract"])

            if "conclusion" in row and pd.notna(row["conclusion"]):
                with st.expander("📌 **Conclusion**", expanded=False):
                    st.write(row["conclusion"])

            # 🎬 NASA MEDIA SECTION - IMPROVED VERSION
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, rgba(0, 150, 255, 0.2) 0%, rgba(0, 100, 200, 0.2) 100%);
                padding: 18px;
                border-radius: 12px;
                border: 2px solid #00d4ff;
                margin: 20px 0;
                box-shadow: 0 4px 15px rgba(0, 200, 255, 0.3);
            ">
                <h4 style="color: #00f5ff; margin: 0 0 8px 0; text-align: center;">
                    🌌 Related NASA Media Gallery
                </h4>
                <p style="color: #e0f7fa; font-size: 0.85em; text-align: center; margin: 0;">
                    Images and videos from NASA's collection
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Extract smart keywords from article
            search_text = f"{row['title']} {row.get('abstract', '')[:200]}"
            keywords = extract_keywords_from_text(search_text, max_keywords=3)
            search_query = " ".join(keywords)
            
            # Try to get images first (more reliable)
            media_results = search_nasa_media(search_query, max_results=3, media_type="image")
            
            # If no images found, try with just the first keyword
            if not media_results and keywords:
                media_results = search_nasa_media(keywords[0], max_results=3, media_type="image")
            
            # Display media gallery
            if media_results:
                st.markdown(f"<p style='color: #e0f7fa; font-size: 0.85em; text-align: center;'>🔍 Search: <strong>{search_query}</strong></p>", unsafe_allow_html=True)
                display_nasa_media_gallery(media_results)
            else:
                st.info(f"🔍 Searching NASA archives for: **{search_query}**")
                st.markdown(f"[🚀 Search manually on NASA Images](https://images.nasa.gov/search?q={search_query.replace(' ', '+')})")
            
            st.markdown("---")
            
            # AI Summary Button
            if st.button(f"🤖 **Summarize Article**", key=f"summarize_{idx}"):
                text_to_summarize = ""
                if "abstract" in row and pd.notna(row["abstract"]):
                    text_to_summarize += f"Abstract:\n{row['abstract']}\n\n"
                if "conclusion" in row and pd.notna(row["conclusion"]):
                    text_to_summarize += f"Conclusion:\n{row['conclusion']}\n\n"
                if text_to_summarize:
                    summary = smart_summarize(text_to_summarize, mode="single")
                    st.markdown(f"""
                    <div style="
                        background: rgba(0, 100, 150, 0.3);
                        border-radius: 10px;
                        padding: 15px;
                        margin: 15px 0;
                        border-left: 4px solid #00FFFF;
                    ">
                        <strong style="color: #00f5ff;">🤖 AI Summary:</strong><br>
                        <span style="color: #e0f7fa;">{summary}</span>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")

        # Summarize All Button
        if len(results) > 1:
            if st.button("🧠 **Generate Comprehensive Summary**", use_container_width=True):
                all_text = ""
                for _, row in results.iterrows():
                    if "abstract" in row and pd.notna(row["abstract"]):
                        all_text += row["abstract"] + "\n\n"
                    if "conclusion" in row and pd.notna(row["conclusion"]):
                        all_text += row["conclusion"] + "\n\n"
                
                if all_text.strip():
                    with st.spinner("🤖 Generating comprehensive summary..."):
                        summary_all = smart_summarize(all_text, max_tokens=800, mode="overview")
                    
                    with st.expander("📋 **Comprehensive Research Overview**", expanded=True):
                        st.markdown(summary_all)

# --- TAB 2: Upload Report (SAMA SEPERTI ASAL) ---
with tabs[1]:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(42, 157, 143, 0.8) 0%, rgba(30, 110, 167, 0.6) 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #2a9d8f;
        box-shadow: 0 0 20px rgba(42, 157, 143, 0.3);
        margin-bottom: 25px;
        text-align: center;
    ">
        <h2 style="color: #2a9d8f; margin: 0;">📑 AI Research Report Analyzer</h2>
        <p style="color: #e0f7fa; margin: 10px 0 0 0;">
            Upload your research report for AI-powered analysis and NASA publication matching
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_upload, col_features = st.columns([2, 1])

    with col_upload:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 25px;
            border-radius: 12px;
            border: 2px dashed rgba(42, 157, 143, 0.5);
            text-align: center;
            margin-bottom: 20px;
        ">
            <h4 style="color: #2a9d8f; margin-top: 0;">📤 Upload Your Research Report</h4>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "**Choose PDF or DOCX File**", 
            type=["pdf", "docx"],
            help="Upload your research paper, thesis, or report for AI analysis"
        )
        
        st.markdown("""
        <div style="
            background: rgba(42, 157, 143, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2a9d8f;
            margin-top: 15px;
        ">
            <p style="color: #e0f7fa; margin: 0; font-size: 0.9em;">
                💡 <strong>Supported formats:</strong> PDF, DOCX<br>
                📝 <strong>Ideal for:</strong> Research papers, literature reviews, thesis chapters<br>
                🔍 <strong>Analysis includes:</strong> Content extraction, similarity matching, gap analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_features:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.08);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(42, 157, 143, 0.3);
            height: 100%;
        ">
            <h4 style="color: #2a9d8f; margin-top: 0;">✨ Analysis Features</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**🔬 Content Extraction**")
        st.markdown("- Full text analysis  \n- Key concept identification")
        
        st.markdown("**📊 Similarity Matching**")
        st.markdown("- NASA publication comparison  \n- Research gap detection")
        
        st.markdown("**🤖 AI Insights**")
        st.markdown("- Summary generation  \n- Relevance scoring  \n- Trend analysis")

    if uploaded_file:
        st.markdown("""
        <div style="
            background: rgba(42, 157, 143, 0.1);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #2a9d8f;
            margin: 20px 0;
        ">
            <h4 style="color: #2a9d8f; margin: 0;">📄 File Analysis in Progress...</h4>
        </div>
        """, unsafe_allow_html=True)
        
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }
        
        col_info, col_status = st.columns(2)
        
        with col_info:
            st.markdown(f"""
            <div style="
                background: rgba(255, 255, 255, 0.05);
                padding: 15px;
                border-radius: 8px;
            ">
                <h5 style="color: #2a9d8f; margin: 0 0 10px 0;">📋 File Information</h5>
                <p style="color: #e0f7fa; margin: 5px 0;">
                    <strong>Name:</strong> {file_details["Filename"]}<br>
                    <strong>Size:</strong> {file_details["File size"]}<br>
                    <strong>Type:</strong> {file_details["File type"]}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_status:
            with st.spinner("🔍 Extracting content and analyzing..."):
                if uploaded_file.type == "application/pdf":
                    report_text = extract_text_from_pdf(uploaded_file)
                else:
                    report_text = extract_text_from_docx(uploaded_file)
            
            st.success("✅ Analysis completed!")
        
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        ">
            <h4 style="color: #2a9d8f; margin: 0 0 15px 0;">📖 Extracted Report Content</h4>
        </div>
        """, unsafe_allow_html=True)
        
        tab_preview, tab_stats = st.tabs(["📝 Content Preview", "📊 Text Statistics"])
        
        with tab_preview:
            st.text_area(
                "**Extracted Text**", 
                report_text[:2500] + "..." if len(report_text) > 2500 else report_text, 
                height=300,
                help="First 2500 characters of extracted content"
            )
        
        with tab_stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Character Count", f"{len(report_text):,}")
            with col2:
                st.metric("Word Count", f"{len(report_text.split()):,}")
            with col3:
                st.metric("Estimated Pages", f"{len(report_text) // 1500 + 1}")
        
        st.markdown("""
        <div style="
            background: rgba(0, 245, 255, 0.1);
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid #00f5ff;
            margin: 20px 0;
        ">
            <h4 style="color: #00f5ff; margin: 0 0 15px 0;">🤖 AI Analysis Results</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔬 Comparing with NASA publications..."):
            comment = ai_comment_on_report(report_text, df["abstract"].fillna("").tolist())
        
        st.markdown(f"""
        <div style="
            background: rgba(0, 100, 150, 0.2);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #00f5ff;
        ">
            <h5 style="color: #00f5ff; margin: 0 0 10px 0;">📈 Relevance Analysis</h5>
            <p style="color: #e0f7fa; margin: 0; line-height: 1.6;">{comment}</p>
        </div>
        """, unsafe_allow_html=True)

# --- Sidebar Graph ---
st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #0d3b66 0%, #1e6ea7 50%, #2a9d8f 100%);
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #00f5ff;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.4);
    margin-bottom: 20px;
">
    <h3 style="color: white; text-align: center; margin: 0; text-shadow: 0 0 10px #00f5ff;">🌌 NASA Graph Visualizations</h3>
    <p style="color: #e0f7fa; text-align: center; font-size: 0.9em;">Explore connections between space research articles</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔗 **Similarity Graph**", use_container_width=True):
        st.session_state.show_similarity_graph = True
        st.session_state.show_knowledge_graph = False

with col2:
    if st.button("🧩 **Knowledge Graph**", use_container_width=True):
        st.session_state.show_knowledge_graph = True
        st.session_state.show_similarity_graph = False

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="
    background: rgba(255, 255, 255, 0.1);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(0, 245, 255, 0.3);
">
    <p style="color: #e0f7fa; font-size: 0.8em; text-align: center; margin: 0;">
        <strong>📊 Graph Features:</strong><br>
        • Content Similarity Analysis<br>
        • Keyword Relationships<br>
        • Research Trends Mapping
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center;">
    <p style="color: #00f5ff; font-size: 2em; margin: 0; text-shadow: 0 0 10px #00f5ff;">608</p>
    <p style="color: #e0f7fa; font-size: 0.9em; margin: 0;">Space Publications</p>
</div>
""", unsafe_allow_html=True)

# Graph Display
st.markdown("---")

if st.session_state.get('show_similarity_graph'):
    st.markdown("### 🔗 Similarity Graph - Article Relationships")
    st.info("🖱️ **Tips**: Drag nodes to explore • Scroll to zoom • Click nodes to see connections")
    with st.spinner("🔄 Generating similarity graph..."):
        display_similarity_graph_online(df)
        
elif st.session_state.get('show_knowledge_graph'):
    st.markdown("### 🧩 Knowledge Graph - Keywords & Concepts") 
    st.info("🖱️ **Tips**: Blue nodes = Articles • Green nodes = Keywords • Drag to explore relationships")
    with st.spinner("🔄 Generating knowledge graph..."):
        display_knowledge_graph_online(df)
else:
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.05);
        padding: 60px 20px;
        border-radius: 15px;
        border: 2px dashed rgba(0, 245, 255, 0.3);
        text-align: center;
        margin: 20px 0;
    ">
        <h3 style="color: #00f5ff;">🌌 Graph Visualization Ready</h3>
        <p style="color: #e0f7fa; font-size: 1.1em;">
            Click on either graph button in the sidebar to visualize NASA research relationships
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
