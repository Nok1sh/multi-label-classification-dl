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
                font-size: 12px;
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
            font-size: 16px;
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