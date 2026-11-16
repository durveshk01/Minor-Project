"""Launch SalesVision with: streamlit run app.py."""

import logging

import streamlit as st

from config.settings import get_settings
from src.database.db_manager import DatabaseManager
from src.ui.dashboard import NAVIGATION, render_page
from src.utils.exceptions import DatabaseError, SalesVisionError
from src.utils.logging_config import configure_logging

st.set_page_config(page_title="SalesVision | Sales Intelligence", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


def main() -> None:
    try:
        secrets = dict(st.secrets) if st.secrets else {}
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        secrets = {}
    try:
        settings = get_settings(secrets)
        settings.ensure_directories()
        configure_logging(settings.root / "outputs/logs")
    except SalesVisionError as exc:
        st.error(str(exc))
        return
    st.markdown(f"<style>{(settings.root / 'assets/dashboard.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="sv-brand"><div class="sv-logo">SV</div><div><div class="sv-brand-name">SalesVision</div><div class="sv-brand-subtitle">Sales intelligence<br/>Forecasting workspace</div></div></div>', unsafe_allow_html=True)
        st.radio("Navigation", NAVIGATION, key="navigation", label_visibility="collapsed")
        st.divider()
        dataset = st.session_state.get("processed")
        if dataset is not None:
            st.caption("ACTIVE DATASET")
            st.write(dataset.metadata.get("name", "User dataset"))
            st.caption(f"{dataset.frequency} · {len(dataset.series):,} periods")
        else:
            st.caption("Load historical sales to begin.")
        st.caption("Academic project · Review 3  \nSales Forecasting Using Python")
    try:
        database = DatabaseManager(settings.database_path)
    except DatabaseError as exc:
        database = None
        st.warning(f"{exc} Analysis, forecasting and report downloads remain available; history saving is unavailable until the path is corrected.")
    try:
        render_page(st.session_state["navigation"], settings, database)
    except SalesVisionError as exc:
        st.error(str(exc))
    except Exception:
        logging.getLogger("salesvision").exception("Unexpected core UI failure on page %s", st.session_state.get("navigation"))
        st.error("This operation could not be completed. Review the selected data/settings and local application logs.")


if __name__ == "__main__":
    main()
