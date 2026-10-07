import pandas as pd
from pathlib import Path

# The folder this script lives in, so it works no matter where you run it from
HERE = Path(__file__).resolve().parent

# Differences under $1 are treated as rounding, not a short payment
TOLERANCE = 1.00


# 1. Load the two files
claims = pd.read_csv(HERE / "claims.csv")
payments = pd.read_csv(HERE / "payments.csv")


# 2. Add up payments per claim
# (a supplier can pay one claim in more than one payment)
paid = payments.groupby("claim_id", as_index=False)["amount_paid"].sum()


# 3. Put claims and payments side by side
# how="left" keeps EVERY claim, even ones with no payment at all
df = claims.merge(paid, on="claim_id", how="left")

# Claims with no payment come through as blank (NaN), so turn those into 0
df["amount_paid"] = df["amount_paid"].fillna(0)


# 4. Work out how much is still owed on each claim
df["owed"] = df["claim_amount"] - df["amount_paid"]


# 5. Give each claim a status
def get_status(row):
    if row["amount_paid"] == 0:
        return "Unpaid"
    elif row["owed"] > TOLERANCE:
        return "Short paid"
    else:
        return "Paid"

df["status"] = df.apply(get_status, axis=1)


# 6. Keep only the claims we need to chase
not_paid = df[df["status"] != "Paid"]


# 7. Show the results and save them
print("Claims to chase:")
print(not_paid[["claim_id", "supplier", "claim_amount", "amount_paid", "owed", "status"]].to_string(index=False))

print()
print(f"Total claimed: ${df['claim_amount'].sum():,.2f}")
print(f"Total paid:    ${df['amount_paid'].sum():,.2f}")
print(f"Still owed:    ${not_paid['owed'].sum():,.2f}")

not_paid.to_csv(HERE / "not_paid.csv", index=False)
print()
print("Saved to not_paid.csv")
