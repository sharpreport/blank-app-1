import streamlit as st
from datetime import date

st.set_page_config(
    page_title="SharpReport NRFI Scanner",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ SharpReport NRFI Scanner")
st.subheader("MLB First-Inning Probability Model")

st.write(f"Date: {date.today().strftime('%B %d, %Y')}")

st.divider()

if st.button("RUN TODAY'S MLB SLATE", type="primary", use_container_width=True):

    st.success("Scanner is running.")

    st.write("Today's MLB games will eventually appear here.")

    st.write(
        """
        Future scanner will automatically retrieve:

        • MLB schedule  
        • Starting pitchers  
        • Confirmed lineups  
        • Statcast statistics  
        • Park factors  
        • Weather  
        • NRFI/YRFI sportsbook prices  
        • Model probabilities  
        • Fair odds  
        • Model edge
        """
    )

st.divider()

st.caption("SharpReport • Data-Driven • Transparent • Consistent")