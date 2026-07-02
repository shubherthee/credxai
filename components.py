import streamlit as st


def load_css(path: str = "styles.css") -> None:
    """Load external CSS into Streamlit."""
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def bank_header() -> None:
    st.markdown(
        """
        <div class="bank-header">
            <div class="brand-stack">
                <div class="bank-logo">CredXAI</div>
                <div class="bank-subtitle">LightGBM-Trained Hybrid XAI Loan Risk Assessment</div>
            </div>
            <div class="header-badge">
                Explainable Credit Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_heading(overline: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-heading">
            <div class="overline">{overline}</div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(value: str, label: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="metric-card metric-{tone}">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def report_section(title: str, subtitle: str | None = None) -> None:
    subtitle_html = f"<div class='section-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="report-section-title">
            <div>{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_banner(decision: str, risk_score: float, threshold: float) -> None:
    if decision == "APPROVED":
        st.markdown(
            f"""
            <div class="decision-approved">
                <div class="decision-title">Application Approved</div>
                <div class="decision-copy">
                    Risk score <strong>{risk_score:.1%}</strong> is within the selected threshold of <strong>{threshold:.0%}</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="decision-rejected">
                <div class="decision-title">Application Declined</div>
                <div class="decision-copy">
                    Risk score <strong>{risk_score:.1%}</strong> exceeds the selected threshold of <strong>{threshold:.0%}</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def comparison_card(actual_status: str, model_decision: str) -> None:
    expected = "Approved" if actual_status == "Fully Paid" else "Rejected"
    predicted = "Approved" if model_decision == "APPROVED" else "Rejected"
    matched = expected == predicted
    badge_class = "match-good" if matched else "match-bad"
    badge_text = "Correct Match" if matched else "Mismatch"

    st.markdown(
        f"""
        <div class="comparison-card">
            <div>
                <div class="comparison-label">Historical Outcome</div>
                <div class="comparison-value">{actual_status}</div>
            </div>
            <div>
                <div class="comparison-label">Model Prediction</div>
                <div class="comparison-value">{predicted}</div>
            </div>
            <div>
                <div class="comparison-label">Validation Result</div>
                <div class="match-badge {badge_class}">{badge_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
