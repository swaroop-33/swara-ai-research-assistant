# Hugging Face Spaces Deployment Guide for SWARA

Follow this step-by-step guide to deploy the SWARA Grounded AI Research Assistant to Hugging Face Spaces.

---

## 1. Space Creation Configuration

When creating a new Space on Hugging Face:
1. **Space Name:** Choose a name (e.g., `swara-ai-assistant`).
2. **License:** Open Source license of your choice (e.g., `mit` or `apache-2.0`).
3. **Select the Space SDK:** Select **Docker** (do not select Streamlit or Gradio).
4. **Choose a Docker Template:** Select **Blank** (do not select template presets).
5. **Space Hardware:** Select the default **CPU Basic (Free - 2 vCPU, 16GB RAM, 50GB Storage)**.
6. **Visibility:**
   * **Recommendation:** **Private** (recommended initially because SWARA does not have an authentication layer. A private space ensures only you can query the system and upload files, protecting your Groq API usage limits).

---

## 2. Environment Variables & Secrets Configuration

Once the Space is created, navigate to **Settings** > **Variables and Secrets** to add the following configurations:

### Secrets (Encrypted Variables)
*   **`GROQ_API_KEY`**: Paste your production Groq API Key (starts with `gsk_`). This must be added as a **Secret** so it is encrypted and hidden from the public.

### Variables (Plaintext Configurations)
*   **`SWARA_API_BASE`**: Set to `http://127.0.0.1:8000/api/v1`. This environment variable tells the Streamlit frontend to make internal API calls to the FastAPI backend running in the same container.
*   **`ENVIRONMENT`**: Set to `production`.

---

## 3. Deployment Steps via GitHub

Hugging Face Spaces are backed by a Git repository. You can push your code directly to Hugging Face or set up a GitHub Actions workflow.

### Option A: Directly pushing to Hugging Face Git Remote
1. In your Hugging Face Space page, copy the remote repository URL (found under **Clone this space**).
2. Open your local terminal in the project directory and run:
   ```bash
   # Add Hugging Face as a remote repository
   git remote add hf https://huggingface.co/spaces/<your-username>/<your-space-name>

   # Push files to Hugging Face (forces deployment start)
   git push -f hf main
   ```

### Option B: Automatic Sync from GitHub
1. Create a repository on GitHub and upload your code.
2. In your Hugging Face Space settings, navigate to **Github Network Connection**.
3. Link your GitHub repository to the Space to trigger builds automatically on every `git push`.

---

## 4. How to Read Build & Container Logs

1. Navigate to your Hugging Face Space page in the browser.
2. At the top right of the Space interface, click the **Logs** tab.
3. **Build Logs:** Displays the Docker build progress. You can see package installations and system dependencies setup.
4. **Container Logs:** Shows the active runtime logs of the running container. This displays the output printed by `start.sh`, Uvicorn (FastAPI), and Streamlit.

---

## 5. Verifying Server Startups

### Verifying FastAPI Startup
Look for the following entries in the **Container Logs**:
```text
=== [1/4] Starting FastAPI ===
INFO:     Started server process [12]
INFO:     Waiting for application startup.
INFO:     Logging initialized | level=INFO | env=production
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
Followed by the health loop outputs:
```text
Waiting for API... (0 seconds elapsed)
=== [3/4] API Ready ===
```

### Verifying Streamlit Startup
Look for these entries at the bottom of the logs indicating successful port binding:
```text
=== [4/4] Starting Streamlit ===
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:7860
  Network URL: http://10.x.x.x:7860
```

---

## 6. Common Deployment Errors & Troubleshooting

### 1. `\r: command not found` or `start.sh: 2: Syntax error`
*   **Cause:** `start.sh` was saved or committed with Windows CRLF line endings instead of Unix LF line endings.
*   **Resolution:** In VS Code, open `start.sh`, click `CRLF` in the bottom status bar, change it to `LF`, save the file, commit, and push again.

### 2. `sqlite3.InitializationError: Chroma requires SQLite 3.35.0 or higher`
*   **Cause:** Outdated SQLite version inside the container.
*   **Resolution:** The custom `Dockerfile` resolves this by using `python:3.11-slim` (built on Debian Bookworm), which includes SQLite `3.40+`. If you modify the base image, ensure it is Bookworm-based.

### 3. `Permission Denied` on running `start.sh`
*   **Cause:** The executable bit is missing on `start.sh`.
*   **Resolution:** Verified by `RUN chmod +x start.sh` inside the `Dockerfile`. Ensure this layer remains intact in your Docker configuration.

### 4. `API Key is missing` / Generation Failures
*   **Cause:** `GROQ_API_KEY` was not configured under Secrets in Space Settings, or the variable name was typed incorrectly.
*   **Resolution:** Verify spelling in HF Settings. Do not write the API key directly into your `.env` or codebase.

---

## 7. Post-Deployment Verification Checklist

Once the space shows a green **Running** status, perform the following validation steps:

*   [ ] **Check App Load:** Verify the SWARA landing page loads successfully without a "Connection Error" screen.
*   [ ] **Upload a PDF:** Upload a small PDF document (e.g. 1-5 pages) using the sidebar. Verify the progress logs:
    *   *Uploading document...*
    *   *Reading document contents...*
    *   *Understanding document knowledge...*
    *   *Preparing SWARA...*
    *   Verify a green checkmark `✓ <filename>` appears under Uploaded Documents.
*   [ ] **Check Session Isolation:**
    1. Ask a question about the uploaded document in Tab A. Confirm you receive a grounded response.
    2. Open a separate Private/Incognito browser window (Tab B) to load the Space. Verify that "Uploaded Documents" is empty and the chat is empty.
    3. Ask a question in Tab B. Confirm it returns a message indicating no documents are uploaded, proving your data is isolated.
*   [ ] **Test New Chat:** In Tab A, click `✨ New Chat`. Verify that the stats metrics reset to zero, the uploaded document disappears from active context, and a new session starts.
*   [ ] **Verify Citations:** Perform a query and expand the "View Sources & Citations" tray. Confirm that snippets match the uploaded document context.
