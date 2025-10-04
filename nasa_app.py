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
