"""
Supplier Rebate ("Payback") Reconciliation
==========================================

Business problem
----------------
An electrical distributor often sells to contractors below its own cost because
a manufacturer agreed to a special price for that job (a "Special Pricing
Agreement", SPA, also called ship-and-debit). The distributor then files a
claim with the manufacturer for the difference:

    claim amount = quantity x (distributor cost - agreed SPA cost)

The manufacturer pays the claim back later as a credit memo. Money leaks when
credits are short, late, never arrive, or when the same claim is filed twice.

What this script does
---------------------
This is step 3 of 3. It reads the cleaned tables in data/clean/, which
clean_data.py produces from the raw exports.

1. Matches credits to claims and computes the variance on each claim.
2. Flags every exception: short paid, unpaid and overdue, duplicates,
   overpaid, late submissions, and credits we cannot match.
3. Ages the open balance and ranks what to chase first.
4. Writes CSV work-lists and a one-page dashboard image.

Run:  python rebate_reconciliation.py
Needs: pandas, numpy, matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "output"
CLEAN = Path(__file__).resolve().parent / "data" / "clean"
AS_OF = pd.Timestamp("2026-09-30")   # reconciliation date
PAYMENT_TERMS_DAYS = 60              # supplier should pay a claim within this
TOLERANCE = 1.00                     # ignore rounding differences under $1


# --------------------------------------------------------------------------
# 1. Load the cleaned tables
# --------------------------------------------------------------------------
def load_clean():
    claims = pd.read_csv(CLEAN / "claims_clean.csv", parse_dates=["invoice_date", "claim_date"])
    credits = pd.read_csv(CLEAN / "credits_clean.csv", parse_dates=["credit_date"])
    terms = pd.read_csv(CLEAN / "supplier_terms.csv")
    return claims, credits, terms


# --------------------------------------------------------------------------
# 2. Reconciliation
# --------------------------------------------------------------------------
def reconcile(claims: pd.DataFrame, credits: pd.DataFrame, terms: pd.DataFrame):
    credits = credits.copy()

    # Credits we cannot match to a claim go on their own research list.
    matched = credits["claim_id"].isin(claims["claim_id"])
    unmatched_credits = credits[~matched]

    paid = (credits[matched].groupby("claim_id")
            .agg(credit_amount=("credit_amount", "sum"),
                 last_credit_date=("credit_date", "max"),
                 credit_memos=("credit_memo", lambda x: ", ".join(x))))

    rec = claims.merge(paid, on="claim_id", how="left")
    rec["credit_amount"] = rec["credit_amount"].fillna(0.0)
    rec["variance"] = (rec["claim_amount"] - rec["credit_amount"]).round(2)
    rec["days_outstanding"] = (AS_OF - rec["claim_date"]).dt.days

    window = rec["supplier"].map(terms.set_index("supplier")["claim_window_days"])
    rec["late_submission"] = (rec["claim_date"] - rec["invoice_date"]).dt.days > window

    # Duplicate = same supplier, invoice, SKU and quantity claimed more than once.
    # The first filing is the real one; later copies are flagged.
    key = ["supplier", "invoice_no", "sku", "qty"]
    rec = rec.sort_values(["claim_date", "claim_id"])
    rec["duplicate"] = rec.duplicated(key, keep="first")

    paid_none = rec["credit_amount"] == 0
    overdue = rec["days_outstanding"] > PAYMENT_TERMS_DAYS
    conditions = [
        rec["duplicate"],
        rec["variance"].abs() <= TOLERANCE,
        rec["variance"] < -TOLERANCE,
        paid_none & rec["late_submission"],
        paid_none & overdue,
        paid_none,
        rec["variance"] > TOLERANCE,
    ]
    labels = ["Duplicate claim", "Paid in full", "Overpaid", "Unpaid - late submission",
              "Unpaid - overdue", "Unpaid - within terms", "Short paid"]
    rec["status"] = np.select(conditions, labels, default="Review")

    # What we can realistically go back to the supplier for.
    recoverable_status = ["Short paid", "Unpaid - overdue"]
    rec["recoverable"] = np.where(rec["status"].isin(recoverable_status), rec["variance"], 0.0)
    # A duplicate the supplier actually paid will be charged back: that is exposure.
    rec["duplicate_exposure"] = np.where(rec["duplicate"], rec["credit_amount"], 0.0)

    rec["aging_bucket"] = pd.cut(rec["days_outstanding"], [-1, 30, 60, 90, 180, 10_000],
                                 labels=["0-30", "31-60", "61-90", "91-180", "180+"])
    return rec.sort_values("claim_id").reset_index(drop=True), unmatched_credits


def summarize(rec, unmatched_credits):
    s = {}
    s["claims_filed"] = len(rec)
    s["claimed_total"] = rec["claim_amount"].sum()
    s["credited_total"] = rec["credit_amount"].sum()
    s["open_balance"] = rec.loc[rec["variance"] > TOLERANCE, "variance"].sum()
    s["recoverable_total"] = rec["recoverable"].sum()
    s["short_paid"] = rec.loc[rec["status"] == "Short paid", "recoverable"].sum()
    s["unpaid_overdue"] = rec.loc[rec["status"] == "Unpaid - overdue", "recoverable"].sum()
    s["pending_within_terms"] = rec.loc[rec["status"] == "Unpaid - within terms", "variance"].sum()
    s["late_submission_lost"] = rec.loc[rec["status"] == "Unpaid - late submission", "variance"].sum()
    s["duplicate_count"] = int(rec["duplicate"].sum())
    s["duplicate_exposure"] = rec["duplicate_exposure"].sum()
    s["unmatched_credit_count"] = len(unmatched_credits)
    s["unmatched_credit_total"] = unmatched_credits["credit_amount"].sum()
    s["recovery_rate"] = s["credited_total"] / s["claimed_total"]
    return s


# --------------------------------------------------------------------------
# 3. Outputs
# --------------------------------------------------------------------------
BLUE = "#2a78d6"
GRAY = "#b5b4ae"
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
AGING_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # older = darker


def money(x):
    return f"${x/1e6:,.2f}M" if abs(x) >= 1e6 else f"${x/1e3:,.0f}K" if abs(x) >= 1e3 else f"${x:,.0f}"


def style(ax, title):
    ax.set_facecolor("#fcfcfb")
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10, fontweight="bold")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9.5)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbar(ax, series, title, colors=BLUE):
    series = series.sort_values()
    bars = ax.barh(series.index, series.values, color=colors, height=0.62)
    style(ax, title)
    ax.xaxis.set_major_formatter(lambda v, _: money(v))
    for b, v in zip(bars, series.values):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, "  " + money(v),
                va="center", fontsize=9.5, color=INK)
    ax.set_xlim(0, series.max() * 1.22)


def dashboard(rec, s, path):
    fig = plt.figure(figsize=(14, 8.6), facecolor="#fcfcfb")
    gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 1.6, 1.6], hspace=0.55, wspace=0.45,
                          left=0.14, right=0.97, top=0.88, bottom=0.07)
    fig.text(0.02, 0.95, "Supplier rebate reconciliation", fontsize=18, fontweight="bold", color=INK)
    fig.text(0.02, 0.915, f"{s['claims_filed']:,} special-pricing claims across {rec['supplier'].nunique()} suppliers, "
             f"as of {AS_OF:%B %d, %Y}  ·  sample data", fontsize=10.5, color=INK_2)

    kpis = [("Claimed", money(s["claimed_total"])),
            ("Credited", f"{money(s['credited_total'])}  ({s['recovery_rate']:.0%})"),
            ("Recoverable now", money(s["recoverable_total"])),
            ("Duplicate exposure", f"{money(s['duplicate_exposure'])}  ({s['duplicate_count']} claims)")]
    kgs = fig.add_gridspec(1, 4, left=0.02, right=0.97, top=0.86, bottom=0.76, wspace=0.15)
    for i, (label, value) in enumerate(kpis):
        ax = fig.add_subplot(kgs[0, i]); ax.axis("off")
        ax.text(0, 0.75, label.upper(), fontsize=9, color=INK_3, fontweight="bold")
        ax.text(0, 0.05, value, fontsize=19 if i != 2 else 21, color=BLUE if i == 2 else INK, fontweight="bold")

    # Where the money is: open dollars by exception type.
    order = ["Short paid", "Unpaid - overdue", "Unpaid - within terms", "Unpaid - late submission"]
    by_status = rec[rec["status"].isin(order)].groupby("status")["variance"].sum().reindex(order)
    by_status.index = ["Short paid", "Unpaid, overdue", "Unpaid, within terms", "Unpaid, filed late"]
    # Blue = chase now; gray = not yet due, or likely lost because it was filed late.
    hbar(fig.add_subplot(gs[1, :2]), by_status, "Open dollars by issue (blue = recoverable now)",
         colors=[GRAY if k in ("Unpaid, within terms", "Unpaid, filed late") else BLUE
                 for k in by_status.sort_values().index])

    # Aging of the open balance.
    ax = fig.add_subplot(gs[1, 2])
    open_ = rec[(rec["variance"] > TOLERANCE) & ~rec["duplicate"]]
    aging = open_.groupby("aging_bucket", observed=False)["variance"].sum()
    bars = ax.bar(aging.index.astype(str), aging.values, color=AGING_RAMP, width=0.66)
    style(ax, "Open balance by age (days)")
    ax.grid(axis="x", visible=False); ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.yaxis.set_major_formatter(lambda v, _: money(v))
    for b, v in zip(bars, aging.values):
        ax.text(b.get_x() + b.get_width() / 2, v, money(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylim(0, aging.max() * 1.18)

    # Who owes us the most.
    by_sup = rec.groupby("supplier")["recoverable"].sum()
    hbar(fig.add_subplot(gs[2, :2]), by_sup, "Recoverable now, by supplier")

    # Supplier payment reliability: share of claimed dollars credited.
    ax = fig.add_subplot(gs[2, 2])
    real = rec[~rec["duplicate"] & (rec["days_outstanding"] > PAYMENT_TERMS_DAYS)]
    rate = (real.groupby("supplier")["credit_amount"].sum() / real.groupby("supplier")["claim_amount"].sum()).sort_values()
    bars = ax.barh([n.split()[0] for n in rate.index], rate.values, color=BLUE, height=0.62)
    style(ax, "Paid rate, claims past terms")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlim(0, 1.15)
    for b, v in zip(bars, rate.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.0%}", va="center", fontsize=9.5, color=INK)

    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    claims, credits, terms = load_clean()
    rec, unmatched = reconcile(claims, credits, terms)
    s = summarize(rec, unmatched)

    rec.to_csv(OUT / "reconciliation_full.csv", index=False)
    # The chase list: what the rebate team should work today, biggest first.
    chase = (rec[rec["recoverable"] > 0]
             .sort_values(["recoverable"], ascending=False)
             [["claim_id", "supplier", "spa_number", "invoice_no", "customer", "sku", "qty",
               "claim_date", "days_outstanding", "claim_amount", "credit_amount", "recoverable", "status"]])
    chase.to_csv(OUT / "chase_list.csv", index=False)
    rec[rec["duplicate"]].to_csv(OUT / "duplicate_claims.csv", index=False)
    unmatched.to_csv(OUT / "unmatched_credits.csv", index=False)
    (rec.groupby("supplier")
        .agg(claims=("claim_id", "count"), claimed=("claim_amount", "sum"), credited=("credit_amount", "sum"),
             recoverable=("recoverable", "sum"), duplicates=("duplicate", "sum"))
        .round(2).sort_values("recoverable", ascending=False)
        .to_csv(OUT / "supplier_summary.csv"))
    dashboard(rec, s, OUT / "dashboard.png")

    lines = [
        f"Rebate reconciliation as of {AS_OF:%Y-%m-%d}",
        f"  Claims filed:              {s['claims_filed']:,}",
        f"  Total claimed:             ${s['claimed_total']:,.2f}",
        f"  Total credited:            ${s['credited_total']:,.2f}  ({s['recovery_rate']:.1%})",
        f"  Recoverable now:           ${s['recoverable_total']:,.2f}",
        f"    short paid:              ${s['short_paid']:,.2f}",
        f"    unpaid past {PAYMENT_TERMS_DAYS} days:     ${s['unpaid_overdue']:,.2f}",
        f"  Pending, within terms:     ${s['pending_within_terms']:,.2f}",
        f"  Likely lost, filed late:   ${s['late_submission_lost']:,.2f}",
        f"  Duplicate claims:          {s['duplicate_count']}  (${s['duplicate_exposure']:,.2f} already paid, chargeback risk)",
        f"  Unmatched credits:         {s['unmatched_credit_count']}  (${s['unmatched_credit_total']:,.2f} to research)",
        "",
        "Claims by status:",
        rec["status"].value_counts().to_string(),
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
