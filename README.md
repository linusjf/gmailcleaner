# Gmail Marketing Email Classifier

Automatically identifies marketing emails in your Gmail inbox using a **Naive Bayes** classifier (scikit-learn) and applies a `Marketing` label — no external AI API required.

---

## 1. Google Cloud Setup (one-time)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or use an existing one)
3. Enable the **Gmail API**: APIs & Services → Enable APIs → search "Gmail API"
4. Create OAuth credentials:
   - APIs & Services → Credentials → Create Credentials → **OAuth client ID**
   - Application type: **Desktop app**
   - Download the JSON file and save it as **`credentials.json`** in this folder
5. Add your Gmail address as a test user under **OAuth consent screen → Test users**

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. First Run (authenticate)

```bash
python classifier.py scan --dry-run
```

A browser window will open for Google OAuth. Sign in and grant access.  
`--dry-run` previews flagged emails without applying any labels.

---

## 4. Apply Labels for Real

```bash
python classifier.py scan
```

Scans up to 50 inbox emails, applies the `Marketing` label to those with ≥75% confidence.

**Adjust confidence threshold** (0–1, lower = more aggressive):
```bash
python classifier.py scan --threshold 0.65
```

---

## 5. Improve the Model (optional but recommended)

The model ships with built-in seed examples. You can teach it from your own emails:

```bash
# Mark text as marketing
python classifier.py label "50% off sale this weekend only shop now" --marketing

# Mark text as not marketing
python classifier.py label "re: project deadline please review the doc" --not-marketing
```

Once you have 20+ examples, retrain:
```bash
python classifier.py retrain
```

---

## Files

| File | Purpose |
|------|---------|
| `classifier.py` | Main script |
| `credentials.json` | Your Google OAuth credentials (you provide) |
| `token.json` | Auto-generated after first login |
| `marketing_model.pkl` | Saved model (auto-created) |
| `training_data.json` | Your labelled examples (grows over time) |

---

## Tips

- Run on a **schedule** (e.g. cron job) to continuously label new emails:
  ```bash
  # Every hour — add to crontab
  0 * * * * cd /path/to/folder && python classifier.py scan
  ```
- Create a **Gmail filter** to auto-archive or delete the `Marketing` label
- The model **gets smarter** the more examples you add via `label` command
