"""
Gmail Marketing Email Classifier
Uses Naive Bayes (scikit-learn) to classify emails and apply a 'Marketing' label.
"""

import os
import json
import pickle
import base64
import re
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
CREDENTIALS_FILE = "credentials.json"   # Downloaded from Google Cloud Console
TOKEN_FILE = "token.json"
MODEL_FILE = "marketing_model.pkl"
DATA_FILE = "training_data.json"
LABEL_NAME = "Marketing"
BATCH_SIZE = 50   # emails to scan per run


# ── Auth ──────────────────────────────────────────────────────────────────────
def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ── Label helpers ─────────────────────────────────────────────────────────────
def get_or_create_label(service, label_name):
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for lbl in labels:
        if lbl["name"].lower() == label_name.lower():
            return lbl["id"]
    new_label = service.users().labels().create(
        userId="me",
        body={
            "name": label_name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        },
    ).execute()
    print(f"✅ Created label '{label_name}'")
    return new_label["id"]


def apply_label(service, msg_id, label_id):
    service.users().messages().modify(
        userId="me",
        id=msg_id,
        body={"addLabelIds": [label_id]},
    ).execute()


# ── Email fetching ────────────────────────────────────────────────────────────
def fetch_emails(service, max_results=BATCH_SIZE, query="in:inbox"):
    messages = []
    resp = service.users().messages().list(
        userId="me", maxResults=max_results, q=query
    ).execute()
    for msg_ref in resp.get("messages", []):
        msg = service.users().messages().get(
            userId="me", id=msg_ref["id"], format="full"
        ).execute()
        messages.append(msg)
    return messages


def extract_text(msg):
    """Extract subject + body text from a Gmail message object."""
    headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
    subject = headers.get("Subject", "")
    sender = headers.get("From", "")
    snippet = msg.get("snippet", "")

    body = ""
    parts = msg["payload"].get("parts", [])
    if not parts:
        data = msg["payload"].get("body", {}).get("data", "")
        if data:
            body = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
    else:
        for part in parts:
            if part["mimeType"] == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    body = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="ignore")
                    break

    # Combine fields — sender domain is a strong signal
    return f"{subject} {sender} {snippet} {body[:500]}"


def clean_text(text):
    text = re.sub(r"http\S+", " url ", text)
    text = re.sub(r"[^a-zA-Z0-9@.\s]", " ", text)
    return text.lower()


# ── Model ─────────────────────────────────────────────────────────────────────
SEED_MARKETING = [
    "unsubscribe offer deal discount sale limited time coupon promo free shipping",
    "newsletter weekly digest update promotional offers exclusive member",
    "click here buy now shop today special offer flash sale clearance",
    "you have been selected winner prize claim reward loyalty points",
    "new arrivals trending now hot deals best sellers top picks",
    "verify your email marketing update terms privacy policy",
    "no longer wish to receive emails manage preferences opt out",
    "black friday cyber monday holiday sale savings deals",
    "product announcement new feature launch early access beta",
    "referral program earn credits invite friends bonus reward",
]

SEED_NOT_MARKETING = [
    "meeting tomorrow agenda please review attached document",
    "re: your question follow up from our conversation",
    "invoice attached payment due account statement",
    "your order has shipped tracking number delivery",
    "password reset security alert login attempt",
    "interview schedule calendar invite accepted",
    "project update status report deadline approaching",
    "family dinner weekend plans catch up soon",
    "ticket #12345 support request resolved",
    "github pull request code review approved merged",
]


def build_seed_model():
    """Create a baseline model from hard-coded seed examples."""
    texts = SEED_MARKETING + SEED_NOT_MARKETING
    labels = [1] * len(SEED_MARKETING) + [0] * len(SEED_NOT_MARKETING)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def load_or_create_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        print(f"📦 Loaded existing model from {MODEL_FILE}")
    else:
        model = build_seed_model()
        save_model(model)
        print("🌱 Created seed model (will improve as you label more emails)")
    return model


def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)


def retrain_model(model):
    """Retrain on accumulated human-labelled data."""
    if not os.path.exists(DATA_FILE):
        print("No labelled training data yet — using current model.")
        return model

    with open(DATA_FILE) as f:
        data = json.load(f)

    if len(data) < 20:
        print(f"Only {len(data)} labelled examples — need 20+ to retrain. Keeping current model.")
        return model

    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    # Add seeds to avoid overfitting on small corpora
    texts += SEED_MARKETING + SEED_NOT_MARKETING
    labels += [1] * len(SEED_MARKETING) + [0] * len(SEED_NOT_MARKETING)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.1)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n📊 Retrained model evaluation:")
    print(classification_report(y_test, y_pred, target_names=["Not Marketing", "Marketing"]))
    save_model(model)
    return model


def add_training_example(text, label):
    """Persist a labelled example for future retraining."""
    data = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            data = json.load(f)
    data.append({"text": clean_text(text), "label": label})
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Main flow ─────────────────────────────────────────────────────────────────
def run_classifier(dry_run=False, confidence_threshold=0.75):
    print("🔐 Authenticating with Gmail...")
    service = get_gmail_service()

    print(f"🏷  Ensuring '{LABEL_NAME}' label exists...")
    label_id = get_or_create_label(service, LABEL_NAME)

    print(f"📬 Fetching up to {BATCH_SIZE} inbox emails...")
    emails = fetch_emails(service)

    model = load_or_create_model()

    flagged, skipped = 0, 0
    for msg in emails:
        raw_text = extract_text(msg)
        text = clean_text(raw_text)
        proba = model.predict_proba([text])[0]
        marketing_prob = proba[1]
        is_marketing = marketing_prob >= confidence_threshold

        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        subject = headers.get("Subject", "(no subject)")[:60]
        sender = headers.get("From", "")[:40]

        if is_marketing:
            flagged += 1
            print(f"  🚩 [{marketing_prob:.0%}] {subject!r} — {sender}")
            if not dry_run:
                apply_label(service, msg["id"], label_id)
        else:
            skipped += 1

    mode = "DRY RUN — " if dry_run else ""
    print(f"\n{mode}✅ Done. Flagged {flagged} / {len(emails)} emails as '{LABEL_NAME}'.")
    if dry_run:
        print("   Run without --dry-run to actually apply labels.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gmail Marketing Email Classifier")
    subparsers = parser.add_subparsers(dest="command")

    # scan
    scan_p = subparsers.add_parser("scan", help="Scan inbox and label marketing emails")
    scan_p.add_argument("--dry-run", action="store_true", help="Preview without applying labels")
    scan_p.add_argument("--threshold", type=float, default=0.75,
                        help="Confidence threshold 0-1 (default 0.75)")

    # retrain
    retrain_p = subparsers.add_parser("retrain", help="Retrain model on labelled data")

    # label (manual feedback)
    label_p = subparsers.add_parser("label", help="Add a training example")
    label_p.add_argument("text", help="Email text to label")
    label_p.add_argument("--marketing", action="store_true", help="Mark as marketing")
    label_p.add_argument("--not-marketing", action="store_true", help="Mark as not marketing")

    args = parser.parse_args()

    if args.command == "scan":
        run_classifier(dry_run=args.dry_run, confidence_threshold=args.threshold)
    elif args.command == "retrain":
        model = load_or_create_model()
        retrain_model(model)
    elif args.command == "label":
        if not (args.marketing or args.not_marketing):
            print("Provide --marketing or --not-marketing")
        else:
            label = 1 if args.marketing else 0
            add_training_example(args.text, label)
            print(f"✅ Saved example (label={'marketing' if label else 'not marketing'})")
    else:
        parser.print_help()
