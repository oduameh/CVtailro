# CVtailro Infrastructure Setup Guide

Complete setup instructions for deploying CVtailro locally and on Railway with Google OAuth, Cloudflare R2 storage, and PostgreSQL.

---

## Table of Contents

1. [Google Cloud Console OAuth Setup](#1-google-cloud-console-oauth-setup)
2. [Cloudflare R2 Storage Setup](#2-cloudflare-r2-storage-setup)
3. [Railway Deployment](#3-railway-deployment)
4. [Local Development](#4-local-development)
5. [Environment Variable Reference](#5-environment-variable-reference)

---

## 1. Google Cloud Console OAuth Setup

CVtailro uses Google OAuth 2.0 to authenticate users and assign admin privileges. Follow these steps to create OAuth credentials.

### 1.1 Create a Google Cloud Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Click the project dropdown at the top of the page (to the right of "Google Cloud").
3. Click **New Project** in the top-right corner of the dialog.
4. Enter the following:
   - **Project name:** `CVtailro`
   - **Organization:** leave as default (or select your organization)
   - **Location:** leave as default
5. Click **Create**.
6. Wait for the project to be created, then select it from the project dropdown.

### 1.2 Configure the OAuth Consent Screen

1. In the left sidebar, navigate to **APIs & Services > OAuth consent screen**.
2. Click **Get started** (or **Configure Consent Screen** if prompted).
3. Select **External** as the user type. Click **Create**.
4. Fill in the required fields:
   - **App name:** `CVtailro`
   - **User support email:** your email address
   - **Developer contact information:** your email address
5. Click **Save and Continue**.
6. On the **Scopes** page, click **Add or Remove Scopes**. Add the following scopes:
   - `openid`
   - `email`
   - `profile`
7. Click **Update**, then **Save and Continue**.
8. On the **Test users** page, optionally add your Gmail address for testing. Click **Save and Continue**.
9. Review the summary and click **Back to Dashboard**.

> **Note:** While in "Testing" mode, only test users you add can log in. To allow any Google account to log in, click **Publish App** on the OAuth consent screen page and confirm. For a small internal tool you can stay in testing mode and add users manually.

### 1.3 Create OAuth 2.0 Credentials

1. In the left sidebar, navigate to **APIs & Services > Credentials**.
2. Click **+ Create Credentials** at the top, then select **OAuth client ID**.
3. Set **Application type** to **Web application**.
4. Set **Name** to `CVtailro Web Client`.
5. Under **Authorized JavaScript origins**, click **+ Add URI** and add:
   - `http://localhost:5050`
   - `https://cvtailro-production.up.railway.app`
6. Under **Authorized redirect URIs**, click **+ Add URI** and add:
   - `http://localhost:5050/auth/google/callback`
   - `https://cvtailro-production.up.railway.app/auth/google/callback`
7. Click **Create**.

### 1.4 Copy Your Credentials

A dialog will appear with your credentials. Copy and save both values securely:

- **Client ID** -- looks like `123456789012-abcdefghijklmnop.apps.googleusercontent.com`
- **Client Secret** -- looks like `GOCSPX-xxxxxxxxxxxxxxxxxxxxxx`

These map to the environment variables:

```
GOOGLE_CLIENT_ID=123456789012-abcdefghijklmnop.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-xxxxxxxxxxxxxxxxxxxxxx
```

> **Important:** If you deploy to a custom domain later, you must return to the Credentials page and add that domain to both the Authorized JavaScript origins and Authorized redirect URIs lists.

---

## 2. Cloudflare R2 Storage Setup

Cloudflare R2 stores generated PDF and markdown artifacts so they persist across deployments and container restarts. R2 is optional -- the application falls back to local file storage when R2 is not configured.

### 2.1 Create a Cloudflare Account

1. Go to [dash.cloudflare.com](https://dash.cloudflare.com/) and sign up (or log in if you have an account).
2. Complete email verification if this is a new account.

### 2.2 Find Your Account ID

1. After logging in, your **Account ID** is displayed on the right sidebar of the main dashboard page, under the **API** section.
2. Alternatively, go to any domain's **Overview** page -- the Account ID is shown in the right sidebar.
3. Copy this value. It looks like a 32-character hex string: `a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4`.

### 2.3 Create an R2 Bucket

1. In the left sidebar, click **R2 Object Storage**.
2. If prompted to activate R2, do so (R2 has a generous free tier: 10 GB storage, 10 million reads/month, 1 million writes/month).
3. Click **Create bucket**.
4. Enter the bucket name: `cvtailro`
5. Select a location hint (choose the region closest to your Railway deployment, e.g., **Automatic** or **Eastern North America**).
6. Click **Create bucket**.

### 2.4 Create an R2 API Token

1. On the R2 overview page, click **Manage R2 API Tokens** (in the right sidebar or top-right area).
2. Click **Create API token**.
3. Configure the token:
   - **Token name:** `CVtailro App`
   - **Permissions:** select **Object Read & Write**
   - **Specify bucket(s):** select **Apply to specific buckets only**, then choose `cvtailro`
   - **TTL:** leave as default (no expiration) or set a long expiration
4. Click **Create API Token**.

### 2.5 Copy Your R2 Credentials

After creation, the token details page will show:

- **Access Key ID** -- looks like a short alphanumeric string
- **Secret Access Key** -- a longer alphanumeric string

> **Warning:** The Secret Access Key is only shown once. Copy it immediately.

These map to the environment variables:

```
R2_ACCOUNT_ID=a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
R2_BUCKET_NAME=cvtailro
```

---

## 3. Railway Deployment

### 3.1 Prerequisites

- A [Railway](https://railway.app/) account (sign up with GitHub recommended)
- Your code pushed to a GitHub repository
- Google OAuth credentials (from Section 1)
- Cloudflare R2 credentials (from Section 2, optional)
- An [OpenRouter](https://openrouter.ai/) API key

### 3.2 Create a Railway Project

1. Log in to [railway.app](https://railway.app/).
2. Click **New Project**.
3. Select **Deploy from GitHub repo**.
4. Authorize Railway to access your GitHub account if prompted.
5. Select the `CVtailro` repository.
6. Railway will detect the `railway.toml` and `Dockerfile` automatically.

### 3.3 Add a PostgreSQL Database

1. Inside your Railway project, click **New** (the "+" button).
2. Select **Database**.
3. Select **Add PostgreSQL**.
4. Railway will provision a PostgreSQL instance and automatically inject the `DATABASE_URL` environment variable into your application service.

> **Verify:** Click on the PostgreSQL service, go to the **Connect** tab, and confirm the `DATABASE_URL` variable is listed. Railway injects this into all linked services automatically.

### 3.4 Set Environment Variables

1. Click on your application service (the one built from your GitHub repo).
2. Go to the **Variables** tab.
3. Click **New Variable** to add each of the following:

| Variable | Value | Notes |
|---|---|---|
| `DATABASE_URL` | *(auto-injected by Railway)* | Comes from the PostgreSQL addon. Do not set manually. If Railway uses `${{Postgres.DATABASE_URL}}` reference syntax, that is fine. |
| `FLASK_APP` | `app.py` | Tells Flask which module to run. |
| `FLASK_SECRET_KEY` | *(generate, see below)* | Used to sign session cookies. Must be a strong random string. |
| `GOOGLE_CLIENT_ID` | *(from Section 1.4)* | Your Google OAuth Client ID. |
| `GOOGLE_CLIENT_SECRET` | *(from Section 1.4)* | Your Google OAuth Client Secret. |
| `ADMIN_EMAILS` | `you@gmail.com` | Comma-separated list of email addresses that receive admin privileges on login. |
| `OPENROUTER_API_KEY` | `sk-or-v1-...` | Your OpenRouter API key. Used as the default API key for all pipeline runs. Can also be set later from the admin panel at `/admin`. |
| `ADMIN_PASSWORD` | *(choose a strong password)* | Password for the `/admin` configuration panel. If not set, the first login to `/admin` will prompt you to create one. |
| `R2_ACCOUNT_ID` | *(from Section 2.5)* | Your Cloudflare Account ID. |
| `R2_ACCESS_KEY_ID` | *(from Section 2.5)* | Your R2 API token Access Key ID. |
| `R2_SECRET_ACCESS_KEY` | *(from Section 2.5)* | Your R2 API token Secret Access Key. |
| `R2_BUCKET_NAME` | `cvtailro` | The name of your R2 bucket. |

#### Generate FLASK_SECRET_KEY

Run this command locally to generate a secure secret key:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output (a 64-character hex string) and paste it as the value for `FLASK_SECRET_KEY`.

### 3.5 Deploy

1. After setting all variables, Railway will automatically trigger a new deployment.
2. Monitor the build logs under the **Deployments** tab.
3. Once deployed, Railway assigns a public URL (e.g., `https://cvtailro-production.up.railway.app`).
4. The health check endpoint at `/api/health` must return HTTP 200 for Railway to consider the deployment healthy.

### 3.6 Post-Deployment Verification

1. Visit `https://your-app.up.railway.app/api/health` -- should return `{"status": "healthy", "backend": "openrouter"}`.
2. Visit the main page and confirm the Google login button works.
3. Log in with the email listed in `ADMIN_EMAILS` and verify you get admin access.
4. Visit `/admin` and confirm the API key is loaded (it will show a masked version).
5. Run a test tailoring job with a sample resume and job description.

### 3.7 Custom Domain (Optional)

If you set up a custom domain on Railway:

1. Go to your application service **Settings** tab in Railway.
2. Under **Networking > Public Networking**, add your custom domain.
3. Return to the [Google Cloud Console Credentials page](https://console.cloud.google.com/apis/credentials) and add your custom domain to:
   - Authorized JavaScript origins: `https://yourdomain.com`
   - Authorized redirect URIs: `https://yourdomain.com/auth/google/callback`

---

## 4. Local Development

### 4.1 Using Docker Compose (Recommended)

Docker Compose starts both the application and a PostgreSQL database.

#### Prerequisites

- Docker and Docker Compose installed
- System libraries for WeasyPrint are handled inside the Docker image

#### Steps

1. Clone the repository and navigate to the project directory:

```bash
cd /Users/emmanuel/Desktop/CVtailro
```

2. Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

3. Edit `.env` with your credentials (see Section 5 for details on each variable). At minimum, set:

```
FLASK_SECRET_KEY=any-random-string-for-local-dev
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
ADMIN_EMAILS=you@gmail.com
OPENROUTER_API_KEY=sk-or-v1-your-key
```

4. Start the services:

```bash
docker-compose up --build
```

5. Open [http://localhost:5050](http://localhost:5050) in your browser.

6. To stop the services:

```bash
docker-compose down
```

7. To stop and remove the database volume (full reset):

```bash
docker-compose down -v
```

> **Note:** The `docker-compose.yml` maps `./output:/app/output` so generated files persist on your host machine. The PostgreSQL data is stored in a named Docker volume (`pgdata`).

### 4.2 Running Without Docker (Python Direct)

When no `DATABASE_URL` is set, the application falls back to a local SQLite database (`cvtailro_dev.db`). R2 storage falls back to local file storage.

#### Prerequisites

- Python 3.13+
- System libraries for WeasyPrint (cairo, pango, gdk-pixbuf). On macOS:

```bash
brew install cairo pango gdk-pixbuf libffi
```

#### Steps

1. Create and activate a virtual environment:

```bash
cd /Users/emmanuel/Desktop/CVtailro
python3 -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables (either export them or create a `.env` file that your shell sources):

```bash
export FLASK_SECRET_KEY="local-dev-secret"
export GOOGLE_CLIENT_ID="your-google-client-id.apps.googleusercontent.com"
export GOOGLE_CLIENT_SECRET="your-google-client-secret"
export ADMIN_EMAILS="you@gmail.com"
export OPENROUTER_API_KEY="sk-or-v1-your-key"
```

4. Run the application:

```bash
python app.py
```

5. Open [http://localhost:5050](http://localhost:5050) in your browser.

> **Note on macOS:** Port 5000 is blocked by AirPlay Receiver. CVtailro uses port 5050 by default. You can override it with the `PORT` environment variable.

### 4.3 Running via CLI (No Web UI)

The orchestrator script runs the pipeline directly from the command line without starting a web server:

```bash
python orchestrator.py \
  --job job_description.txt \
  --resume resume.pdf \
  --api-key sk-or-v1-your-key \
  --model openai/gpt-4o
```

Or set the API key as an environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key"
python orchestrator.py --job job.txt --resume resume.pdf
```

---

## 5. Environment Variable Reference

Complete list of all environment variables used by CVtailro.

| Variable | Required | Default | Description |
|---|---|---|---|
| `FLASK_APP` | No | `app.py` | Flask application entry point. Set this on Railway. |
| `FLASK_SECRET_KEY` | **Yes** | `cvtailro-dev-secret-change-in-production` | Secret key for signing session cookies. **Must** be a strong random string in production. Generate with: `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `DATABASE_URL` | No | `sqlite:///cvtailro_dev.db` | Database connection string. Auto-injected by Railway when PostgreSQL addon is added. Format: `postgresql://user:password@host:port/dbname`. Falls back to SQLite for local development. |
| `GOOGLE_CLIENT_ID` | **Yes** | *(empty)* | Google OAuth 2.0 Client ID from the Google Cloud Console. |
| `GOOGLE_CLIENT_SECRET` | **Yes** | *(empty)* | Google OAuth 2.0 Client Secret from the Google Cloud Console. |
| `ADMIN_EMAILS` | **Yes** | *(empty)* | Comma-separated list of Gmail addresses that receive admin privileges upon login. Example: `alice@gmail.com,bob@gmail.com` |
| `OPENROUTER_API_KEY` | **Yes** | *(empty)* | OpenRouter API key. Used as the default API key for the LLM pipeline. Can also be configured via the admin panel at `/admin`. |
| `ADMIN_PASSWORD` | No | *(none)* | Password for the `/admin` configuration panel. If not set via env var, the first visit to `/admin` will prompt you to set one interactively. |
| `R2_ACCOUNT_ID` | No | *(empty)* | Cloudflare Account ID for R2 storage. Required only if using R2. |
| `R2_ACCESS_KEY_ID` | No | *(empty)* | Cloudflare R2 API token Access Key ID. Required only if using R2. |
| `R2_SECRET_ACCESS_KEY` | No | *(empty)* | Cloudflare R2 API token Secret Access Key. Required only if using R2. |
| `R2_BUCKET_NAME` | No | `cvtailro` | Name of the R2 bucket. Only relevant if R2 credentials are set. |
| `PORT` | No | `5050` | Port the Flask server listens on. Railway sets this automatically. |

### Required vs Optional

- **For production (Railway):** `FLASK_SECRET_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `ADMIN_EMAILS`, and `OPENROUTER_API_KEY` are required. `DATABASE_URL` is auto-injected. R2 variables are recommended but optional.
- **For local development:** `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are required for OAuth login to work. `OPENROUTER_API_KEY` is required to run the pipeline. Everything else has sensible defaults.
- **For CLI usage (orchestrator.py):** Only `OPENROUTER_API_KEY` is needed (or pass `--api-key` flag).
