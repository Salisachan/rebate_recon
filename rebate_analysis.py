"""Check supplier rebate claims against payments using sample CSV files."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Find the folder where this script is saved
HERE = Path(__file__).resolve().parent

# Create a folder to save the results
RESULT = HERE / "result"
RESULT.mkdir(exist_ok=True)

# business rules
TOLERANCE = 1.00
OVERDUE_DAYS = 60
AS_OF = pd.Timestamp("2026-10-09") 

# 1. Read the two CSV files
claims = pd.read_csv(HERE / "claims.csv")
payments = pd.read_csv(HERE / "payments.csv")


# 2. Clean the data before matching anything
claims["claim_id"] = claims["claim_id"].astype("string").str.strip().str.upper()
payments["claim_id"] = payments["claim_id"].astype("string").str.strip().str.upper()
claims["supplier"] = claims["supplier"].astype("string").str.strip()

claims["claim_amount"] = pd.to_numeric(claims["claim_amount"], errors="coerce")
payments["amount_paid"] = pd.to_numeric(payments["amount_paid"], errors="coerce")
claims["claim_date"] = pd.to_datetime(claims["claim_date"], errors="coerce")
payments["payment_date"] = pd.to_datetime(payments["payment_date"], errors="coerce")

# Remove incomplete claims and payments
claims = claims.dropna(subset=["claim_id", "supplier", "claim_date", "claim_amount"])
payments = payments.dropna(subset=["claim_id", "payment_date", "amount_paid"])

# Remove repeated claim rows from the export (not payment rows)
claims = claims.drop_duplicates()

# A claim ID should only appear once in the claims file
if claims["claim_id"].duplicated().any():
    raise ValueError("Some claim IDs appear twice. Check claims.csv before continuing.")


# 3. Add together payments for the same claim
paid = payments.groupby("claim_id", as_index=False)["amount_paid"].sum()

# Keep all claims, including those without a payment
results = claims.merge(paid, on="claim_id", how="left")
results["amount_paid"] = results["amount_paid"].fillna(0)


# 4. Calculate the amount still owed
results["owed"] = (results["claim_amount"] - results["amount_paid"]).round(2)


def get_status(row):
    if row["amount_paid"] == 0:
        return "Unpaid"
    elif row["owed"] > TOLERANCE:
        return "Short paid"
    elif row["owed"] < -TOLERANCE:
        return "Overpaid"
    else:
        return "Paid"


results["status"] = results.apply(get_status, axis=1)

# Flag old claims that still have money owing
results["days_old"] = (AS_OF - results["claim_date"]).dt.days
results["overdue"] = (results["days_old"] > OVERDUE_DAYS) & (results["owed"] > TOLERANCE)


# 5. Make a list of claims suppliers still need to pay
not_paid = results[results["status"].isin(["Unpaid", "Short paid"])].copy()
not_paid = not_paid.sort_values("owed", ascending=False)

print("Claims to follow up:")
print(not_paid[["claim_id", "supplier", "owed", "status", "overdue"]].to_string(index=False))
print()
print(f"Total claimed:     ${results['claim_amount'].sum():,.2f}")
print(f"Total received:    ${results['amount_paid'].sum():,.2f}")
print(f"Needs follow-up:   ${not_paid['owed'].sum():,.2f}")
print(f"Overdue claims:    {not_paid['overdue'].sum()}")

# Save the results in the result folder
results.to_csv(RESULT / "reconciliation_results.csv", index=False)
not_paid.to_csv(RESULT / "not_paid.csv", index=False)

# 6. Show how much each supplier still owes
by_supplier = not_paid.groupby("supplier")["owed"].sum().sort_values()

if not by_supplier.empty:
    plt.figure(figsize=(9, 5))
    plt.barh(by_supplier.index, by_supplier.values)
    plt.title("Outstanding rebate claims by supplier")
    plt.xlabel("Amount owed ($)")
    plt.tight_layout()
    plt.savefig(RESULT / "owed_by_supplier.png")
    plt.close()
    print("Saved chart to owed_by_supplier.png")
else:
    print("No unpaid claims to chart.")

print("Saved results in the result folder")
