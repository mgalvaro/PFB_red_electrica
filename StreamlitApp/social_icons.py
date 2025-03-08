import streamlit as st

def social_icons(linkedin_user, github_user):
    linkedin_url = f"https://www.linkedin.com/in/{linkedin_user}"
    github_url = f"https://github.com/{github_user}"

    st.markdown(
        f"""
        <style>
            .social-icons {{
                display: flex;
                justify-content: left;
                gap: 20px;
            }}
            .social-icons img {{
                width: 20px;
                transition: filter 0.3s ease;
            }}
            @media (prefers-color-scheme: dark) {{
                .social-icons img.github {{
                    filter: invert(1);
                }}
            }}
        </style>

        <div class="social-icons">
            <a href="{linkedin_url}" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png">
            </a>
            <a href="{github_url}" target="_blank">
                <img class="github" src="https://cdn-icons-png.flaticon.com/512/25/25231.png">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

