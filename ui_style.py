import streamlit as st


def colored_progress(
    value: float,
    bar_color="#46B7C6",
    bg_color="#555454"
):
    percent = int(value * 100)

    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border-radius: 10px;
            width: 100%;
            height: 22px;
            margin-bottom: 6px;
        ">
            <div style="
                width: {percent}%;
                height: 100%;
                background: {bar_color};
                border-radius: 10px;
                transition: width 0.4s ease;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 6px;
                color: white;
                font-size: 20px;
                font-weight: 500;
            ">
                {percent}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def colored_button():
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 20px !important;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #45a049;
            color: white;
        }

        .stButton > button:active {
            background-color: #3d8b40;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

def text_styles():
    st.markdown(
        """
        <style>
        .text-title {
            font-family: 'Georgia', serif;
            font-size: 28px;
            font-weight: bold;
            color: #2E8B57;
            margin-bottom: 10px;
        }

        .text-subtitle {
            font-family: 'Arial', sans-serif;
            font-size: 24px !important;
            color: #444444;
            margin-bottom: 8px;
        }

        .text-body {
            font-family: 'Arial', sans-serif;
            font-size: 24px !important;
            color: #333333;
            line-height: 1.5;
            margin-bottom: 2px !important;
        }

        .text-caption {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            color: #888888;
        }

                </style>
        """,
        unsafe_allow_html=True
    )

def select_box_style():
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] > div > div {
            font-size: 20px !important;
            font-family: Arial, sans-serif !important;
        }

        div[data-testid="stSelectbox"] .stSelectbox div[data-baseweb="select"] span {
            font-size: 20px !important;
        }

        div[data-testid="stSelectbox"] .stSelectbox div[role="listbox"] div {
            font-size: 18px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
