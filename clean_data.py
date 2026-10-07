"""
Step 2 of 3: clean the raw extracts.

Reads the raw exports in data/ and fixes the usual problems:
  - blank lines and rows the export repeated
  - supplier names written different ways
  - mixed date formats
  - amounts stored as text with "$" and commas, credits exported as negatives
  - claim references suppliers type many ways ("100123", "clm 100123", ...)

Writes the cleaned tables to data/clean/ and a count of every fix to
output/cleaning_log.txt. The analysis (rebate_reconciliation.py) only reads
data/clean/, so it never has to deal with messy data.

Run:  python clean_data.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
CLEAN = DATA / "clean"
OUT = Path(__file__).resolve().parent / "output"


def load_data():
    """Read the claims register, supplier credit report and supplier terms as-is.
    Everything is read as text so nothing is guessed before cleaning."""
    claims = pd.read_csv(DATA / "claims_submitted.csv", dtype=str)
    credits = pd.read_csv(DATA / "supplier_credits.csv", dtype=str)
    terms = pd.read_csv(DATA / "supplier_terms.csv")
    return claims, credits, terms


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def supplier_key(name: pd.Series) -> pd.Series:
    """'  APEX BREAKERS and CONTROLS ' and 'Apex Breakers & Controls' -> 'APEXBREAKERSANDCONTROLS'."""
    return (name.astype(str).str.upper().str.replace("&", "AND", regex=False)
            .str.replace(r"[^A-Z0-9]", "", regex=True))


def to_money(x: pd.Series) -> pd.Series:
    """'$1,234.50', '1,234.50', '-1234.50' -> 1234.5 (credits can come in negative)."""
    return pd.to_numeric(x.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce").abs()


def normalize_ref(ref: pd.Series) -> pd.Series:
    """'clm 100123', 'CLM100123', '100123', ' clm-100123 ' -> 'CLM-100123'."""
    digits = ref.astype(str).str.extract(r"(\d{6})", expand=False)
    is_claim = ~ref.astype(str).str.upper().str.contains("REB")
    return ("CLM-" + digits).where(is_claim & digits.notna())


def clean(claims, credits, terms):
    """Standardize the raw extracts. Returns cleaned tables and a log of fixes."""
    log = []
    official = dict(zip(supplier_key(terms["supplier"]), terms["supplier"]))

    def common(df, name, key_col, money_cols, date_cols):
        n0 = len(df)
        df = df.dropna(how="all")
        log.append(f"{name}: removed {n0 - len(df)} blank rows")
        n1 = len(df)
        df = df.drop_duplicates(subset=[key_col], keep="first")
        log.append(f"{name}: removed {n1 - len(df)} repeated export rows (same {key_col})")
        fixed = (df["supplier"] != supplier_key(df["supplier"]).map(official)).sum()
        df["supplier"] = supplier_key(df["supplier"]).map(official)
        log.append(f"{name}: standardized {fixed} supplier names "
                   f"({df['supplier'].isna().sum()} could not be mapped)")
        for c in date_cols:
            iso = df[c].str.match(r"^\d{4}-\d{2}-\d{2}$").sum()
            df[c] = pd.to_datetime(df[c], format="mixed")
            log.append(f"{name}: converted {len(df) - iso} {c} values from other date formats")
        for c in money_cols:
            text = (~df[c].str.match(r"^-?\d+(\.\d+)?$")).sum()
            neg = df[c].str.startswith("-").sum()
            df[c] = to_money(df[c])
            log.append(f"{name}: cleaned {text} {c} values stored with $ or commas"
                       + (f", flipped {neg} negative credits to positive" if neg else ""))
        return df.reset_index(drop=True)

    claims = common(claims, "claims", "claim_id", ["distributor_cost", "spa_cost", "claim_amount"],
                    ["invoice_date", "claim_date"])
    claims["qty"] = claims["qty"].astype(int)
    credits = common(credits, "credits", "credit_memo", ["credit_amount"], ["credit_date"])

    credits["claim_id"] = normalize_ref(credits["claim_ref"])
    changed = (credits["claim_ref"] != credits["claim_id"]).sum()
    log.append(f"credits: rewrote {changed} claim references into the standard CLM-###### form "
               f"({credits['claim_id'].isna().sum()} are not claim numbers at all)")
    return claims, credits, log



def main():
    CLEAN.mkdir(parents=True, exist_ok=True); OUT.mkdir(exist_ok=True)
    claims, credits, terms = load_data()
    claims, credits, log = clean(claims, credits, terms)
    claims.to_csv(CLEAN / "claims_clean.csv", index=False)
    credits.to_csv(CLEAN / "credits_clean.csv", index=False)
    terms.to_csv(CLEAN / "supplier_terms.csv", index=False)
    (OUT / "cleaning_log.txt").write_text("\n".join(log) + "\n")
    print("Cleaning:\n  " + "\n  ".join(log))
    print(f"\nWrote {len(claims):,} claims and {len(credits):,} credits to {CLEAN}")


if __name__ == "__main__":
    main()
