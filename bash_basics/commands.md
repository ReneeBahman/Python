pwd                 # 1. Where am I? (print working directory)
ls                  # 2. List files here
ls -l               # 3. List files with details (permissions, size, etc.)
cd ..               # 4. Go up one folder
cd foldername       # 5. Go into a folder
mkdir practice      # 6. Make a new folder
cd practice
echo "hello world" > hello.txt   # 7. Create file with text
cat hello.txt       # 8. Show file contents
mv hello.txt hi.txt # 9. Rename file
rm hi.txt           # 10. Delete file


# Bash Basics — API vs Direct Download Cheat‑Sheet

This cheat‑sheet summarizes the two common ways of pulling data into your projects: using an **API** or a **direct file URL**.

---

## 1. API Download (Authenticated)

When a service provides an **API**, you usually need an **API key/token**.

**Generic Bash example:**

```bash
curl -H "Authorization: Bearer <YOUR_API_TOKEN>" \
     "https://api.example.com/data/endpoint" \
     -o data/raw/mydata.json
```

**Explanation:**

* `curl` → command line tool to fetch data from the web
* `-H` → adds a header to the request (here for authentication)
* `Bearer <YOUR_API_TOKEN>` → standard way to pass your secret key
* final URL → API endpoint (varies by provider)
* `-o` → save output to a file locally

**Example — Kaggle (specialized API):**

```bash
kaggle datasets download -d username/dataset-name -p data/raw --unzip
```

---

## 2. Direct File Download (No Authentication)

When a dataset is hosted as a public file (CSV, JSON, ZIP, etc.), you can grab it directly.

**Using `wget`:**

```bash
wget -P data/raw https://www.renee.com/random_data.csv
```

**Using `curl`:**

```bash
curl -L "https://www.renee.com/random_data.csv" -o data/raw/random_data.csv
```

**Using Python (pandas):**

```python
import pandas as pd

url = "https://www.renee.com/random_data.csv"
df = pd.read_csv(url)
df.to_csv("data/raw/random_data.csv", index=False)
```

---

## Quick Rules of Thumb

* If you see **API token / key** in documentation → use the API + `curl`.
* If you see a **direct file link** (ends in .csv, .json, .zip) → download directly with `wget`, `curl`, or pandas.
* Keep secrets (API keys) **out of Git repos** → store in `~/.config/`, `.env`, or `~/.kaggle/` depending on service.

---