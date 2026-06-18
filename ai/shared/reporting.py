"""
Shared HTML-report helpers
=========================
Small utilities reused across the report builders (eda_report, county_reports,
market_brief, market_narrative) so the matplotlib->base64, percent formatting,
and narrative->HTML snippets live in one place instead of being copy-pasted.
"""

import base64
from io import BytesIO


def fig_to_b64(fig, close=True, **savefig_kwargs):
    """Render a matplotlib figure to a base64 PNG string (for inline <img>)."""
    import matplotlib.pyplot as plt
    kwargs = {"format": "png", "bbox_inches": "tight"}
    kwargs.update(savefig_kwargs)
    buf = BytesIO()
    fig.savefig(buf, **kwargs)
    if close:
        plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fmt_pct(x, none_text="—"):
    """Signed percent string: 1.4 -> '+1.4%', -1.5 -> '-1.5%', None -> none_text."""
    if x is None:
        return none_text
    return f"+{x}%" if x >= 0 else f"{x}%"


def summary_to_paras(summary):
    """A double-newline-delimited summary -> a run of <p>…</p> blocks."""
    return "".join(f"<p>{p}</p>" for p in str(summary).split("\n\n") if p.strip())


def watch_to_li(items):
    """A list of 'watch' bullets -> a run of <li>…</li> elements."""
    return "".join(f"<li>{w}</li>" for w in (items or []))
