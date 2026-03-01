from datasets import load_dataset
import pandas as pd

print("Loading CLINC150 dataset...")

# Load dataset
dataset = load_dataset("clinc_oos", "plus")

train_data = dataset["train"]

# Extract intent names (CLINC stores intents as numeric labels)
intent_names = train_data.features["intent"].names

print("Dataset loaded successfully")


# -----------------------------
# Debug Information (Important)
# -----------------------------

print("\nExample Record:")
print(train_data[0])

print("\nTotal Available Intents:", len(intent_names))

print("\nFirst 20 Intent Names:")
print(intent_names[:])


# -----------------------------
# Intent Mapping (Custom Dataset)
# -----------------------------

intent_map = {

    "order_status": "order_status",
    "order": "order_status",

    "cancel": "cancel_order",
    "cancel_reservation": "cancel_order",

    "report_fraud": "refund_request",
    "transactions": "refund_request",

    "pay_bill": "payment_issue",
    "bill_balance": "payment_issue",
    "card_declined": "payment_issue",

    "bill_due": "subscription_issue",
    "apr": "subscription_issue",

    "lost_luggage": "shipping_issue",
    "replacement_card_duration": "shipping_issue",

    "damaged_card": "return_item",
    "new_card": "return_item",

    "account_blocked": "complaint",
    "freeze_account": "complaint",

    "greeting": "greeting",

    "goodbye": "goodbye"

}


# -----------------------------
# Extract Required Samples
# -----------------------------

filtered_data = []

for item in train_data:

    intent_id = item["intent"]

    intent_name = intent_names[intent_id]

    if intent_name in intent_map:

        filtered_data.append({

            "text": item["text"],

            "intent": intent_map[intent_name]

        })


print("\nFiltered samples:", len(filtered_data))
print("Sample after filtering:")
print(filtered_data[0])



# -----------------------------
# Convert to DataFrame
# -----------------------------

df = pd.DataFrame(filtered_data)
print(df["intent"])

if len(df) == 0:

    raise Exception(
        "No samples found. Check intent_map names."
    )


print("\nOriginal Intent Distribution:")

print(df["intent"].value_counts())


# -----------------------------
# Balance Dataset
# -----------------------------

SAMPLES_PER_INTENT = 30

df_balanced = df.groupby("intent").sample(
    n=SAMPLES_PER_INTENT,
    random_state=42
)


# print("\nBalanced Dataset Distribution:")

# print(df_balanced["intent"].value_counts())


# -----------------------------
# Dataset Validation
# -----------------------------

print("\nValidating dataset...")
unique_intents = df_balanced["intent"].unique()

if len(unique_intents) != 10:

    raise Exception(
        "Dataset must contain exactly 10 intents"
    )


if df_balanced.shape[0] != 300:

    raise Exception(
        "Dataset must contain exactly 300 samples"
    )


print("Dataset validation passed")


# # -----------------------------
# # Save Dataset
# # -----------------------------

OUTPUT_PATH = "datasets/intent_dataset.csv"

df_balanced.to_csv(

    OUTPUT_PATH,

    index=False

)


print("\nDataset saved successfully!")

print("Location:", OUTPUT_PATH)


# -----------------------------
# Preview Dataset
# -----------------------------

print("\nSample Data:")

print(df_balanced.head())