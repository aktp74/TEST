# =========================
# 🎵 Background Music Setup (mute autoplay → unmute selepas klik)
# =========================
music_file = "background.mp3"
if os.path.exists(music_file):
    with open(music_file, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()

    md_audio = f"""
    <script>
    var audio = new Audio("data:audio/mp3;base64,{b64}");
    audio.loop = true;
    audio.autoplay = true;
    audio.volume = 0.15;
    audio.muted = true;  // autoplay dibenarkan kerana mute

    // Unmute & play selepas user klik sekali
    document.addEventListener('click', function() {{
        if (audio.muted) {{
            audio.muted = false;
            audio.play().catch(function(err) {{
                console.log("Play failed:", err);
            }});
        }}
    }});
    </script>
    """

    st.markdown(md_audio, unsafe_allow_html=True)
