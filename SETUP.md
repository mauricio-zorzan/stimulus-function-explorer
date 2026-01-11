# Complete Setup Guide

This guide will walk you through the complete setup process for the Stimulus Function Explorer, from initial installation to syncing functions and standards.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Installation](#initial-installation)
3. [Environment Configuration](#environment-configuration)
4. [Repository Setup](#repository-setup)
5. [Running Tests to Generate Images](#running-tests-to-generate-images)
6. [Syncing Functions and Images](#syncing-functions-and-images)
7. [Syncing Educational Standards](#syncing-educational-standards)
8. [Running the Application](#running-the-application)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **pip** package manager
- **Git** installed
- **pytest** (will be installed with requirements)
- **MySQL database access** (optional, for educational standards)
- **GitHub access** (optional, for repository syncing)

---

## Initial Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mauricio-zorzan/stimulus-function-explorer.git
cd stimulus-function-explorer
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web framework)
- mysql-connector-python (database access)
- requests (API calls)
- python-dotenv (environment variables)
- And other required packages

---

## Environment Configuration

Create a `.env` file in the project root with your credentials:

```bash
# Create .env file
touch .env
```

Add the following variables (all optional, but recommended):

```env
# Database credentials (for educational standards)
DB_USERNAME=your_username
DB_PASSWORD=your_password

# GitHub token (for repository syncing - optional)
GITHUB_TOKEN=your_github_token

# OpenAI API key (for AI search - optional)
OPENAI_API_KEY=your_openai_key
```

**Note:** 
- Database credentials are required for syncing educational standards
- GitHub token is optional - you can set up the repository using SSH keys instead
- OpenAI API key is optional - only needed for AI-powered search features

---

## Repository Setup

The app requires access to the `coach-bot-external-content-generators` repository to discover functions and retrieve test images.

### Option 1: Using the Setup Script (Recommended)

```bash
./setup_repo_without_token.sh
```

This script will:
- Check if the repository is already cloned
- Clone it using SSH (if SSH keys are set up) or HTTPS
- Set up the repository in the `coach-bot-repo/` directory

### Option 2: Manual Setup with SSH Keys

If you have SSH keys set up with GitHub:

```bash
# Clone using SSH
git clone git@github.com:trilogy-group/coach-bot-external-content-generators.git coach-bot-repo

# To sync later
cd coach-bot-repo && git pull
```

**To set up SSH keys (if needed):**
1. Check if you have SSH keys: `ls -la ~/.ssh/id_*.pub`
2. If not, generate one: `ssh-keygen -t ed25519 -C "your_email@example.com"`
3. Add to GitHub: https://github.com/settings/keys
4. Test: `ssh -T git@github.com`

### Option 3: Manual Setup with GitHub CLI

If you have GitHub CLI installed:

```bash
# Authenticate
gh auth login

# Clone
gh repo clone trilogy-group/coach-bot-external-content-generators coach-bot-repo
```

### Option 4: Manual Setup with HTTPS (Requires Token)

If you have a GitHub token:

```bash
# Clone using token
git clone https://YOUR_TOKEN@github.com/trilogy-group/coach-bot-external-content-generators.git coach-bot-repo
```

### Verify Repository Setup

After cloning, verify the repository is set up correctly:

```bash
# Check if repository exists
ls -la coach-bot-repo/

# Check repository status
cd coach-bot-repo && git status && cd ..
```

---

## Running Tests to Generate Images

**Important:** Before syncing functions, you must run the tests to generate images. The sync process scans these generated images but does not run tests automatically.

### Step 1: Navigate to Repository

```bash
cd coach-bot-repo
```

### Step 2: Install Repository Dependencies

The repository may have its own requirements. Check for a `requirements.txt` or `pyproject.toml`:

```bash
# If using pip
pip install -r requirements.txt

# If using poetry (if pyproject.toml exists)
poetry install
```

### Step 3: Run All Drawing Function Tests

```bash
# Run all drawing function tests
pytest src/content_generators/additional_content/stimulus_image/drawing_functions/tests/ -v

# This will generate images in: coach-bot-repo/content/tests/
```

### Step 4: Verify Images Were Generated

```bash
# Check if images were created
ls -la content/tests/*.webp | head -20

# Count total images
ls -1 content/tests/*.webp | wc -l
```

**Expected Result:** You should see multiple `.webp` files in the `content/tests/` directory.

### Step 5: Return to Project Root

```bash
cd ..
```

### Running Specific Tests

If you only want to test specific functions:

```bash
cd coach-bot-repo

# Run tests for a specific file
pytest src/content_generators/additional_content/stimulus_image/drawing_functions/tests/test_base_ten_blocks.py -v

# Run tests matching a pattern
pytest src/content_generators/additional_content/stimulus_image/drawing_functions/tests/ -k "spinner" -v
```

---

## Syncing Functions and Images

Now that tests have generated images, you can sync them into the app.

### Option 1: Using the App (Recommended)

1. **Start the Streamlit app:**
   ```bash
   streamlit run app_new.py
   ```

2. **Click "📦 Sync Repository & Functions"** in the sidebar

3. **Wait for sync to complete** - The app will:
   - Update the repository to the latest version
   - Extract filename patterns from functions
   - Parse test files to identify tested functions
   - Scan and match test images to functions
   - Copy images to `data/images/`
   - Update function metadata
   - Refresh the display

### Option 2: Using Command Line

```bash
# Sync repository first
./sync_coach_bot.sh

# Then run the image retrieval
python3 retrieve_function_images.py --mode test-images
```

### What Gets Synced

- **Function Discovery**: All functions from `coach-bot-repo/src/content_generators/additional_content/stimulus_image/drawing_functions/main.py`
- **Pattern Extraction**: Filename patterns automatically extracted from function code
- **Test Images**: Images from `coach-bot-repo/content/tests/` matched to functions
- **Function Metadata**: JSON files created in `data/functions/` for each function
- **Master Index**: `data/index.json` updated with all functions

### Verifying Sync

After syncing, verify everything worked:

```bash
# Check function files
ls -1 data/functions/*.json | wc -l

# Check images
ls -1 data/images/*.webp | wc -l

# Check index
cat data/index.json | jq '.metadata.total_functions'
```

---

## Syncing Educational Standards

Educational standards are retrieved from a MySQL database and linked to functions.

### Prerequisites

- Database credentials in `.env` file:
  ```env
  DB_USERNAME=your_username
  DB_PASSWORD=your_password
  ```
- Database connection to the standards database

### Option 1: Using the App (Recommended)

1. **Ensure database credentials are in `.env`**

2. **In the Streamlit app, click "📚 Sync Standards"** in the sidebar

3. **Wait for sync to complete** - The app will:
   - Connect to the database
   - Query standards for all functions
   - Retrieve stimulus type specifications
   - Update function JSON files with standards data
   - Refresh the display

### Option 2: Using Command Line

```bash
# Update standards for all functions
python3 update_function_standards.py
```

This uses concurrent processing (10 workers) for faster updates.

### What Gets Synced

- **Educational Standards**: CCSS and other standards linked to functions
- **Stimulus Type Specifications**: Detailed specifications for each standard
- **Function Updates**: Standards added to each function's JSON file
- **Last Updated Timestamp**: Recorded in each function file

### Verifying Standards Sync

After syncing, check a function file:

```bash
# View standards for a specific function
cat data/functions/generate_spinner_with_shapes.json | jq '.educational_standards'
```

---

## Running the Application

### Start the App

```bash
streamlit run app_new.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Browse Functions**: View all functions in the gallery on the home page
2. **Search**: Use the search bar to find functions by name, description, or standards
3. **View Details**: Click on any function to see detailed information, images, and standards
4. **Sync Data**: Use the sync buttons in the sidebar to update functions and standards

### Sync Buttons

- **📦 Sync Repository & Functions**: Updates repository, extracts patterns, scans test images, and organizes functions
- **📚 Sync Standards**: Updates educational standards from the database

---

## Troubleshooting

### Repository Issues

**Problem:** Repository won't clone
- **Solution:** Check your GitHub authentication (SSH keys or token)
- **Alternative:** Use `SETUP_WITHOUT_TOKEN.md` for manual setup options

**Problem:** Repository is out of date
- **Solution:** Run `./sync_coach_bot.sh` or `cd coach-bot-repo && git pull`

### Test Image Issues

**Problem:** No images showing up after sync
- **Check:** Did you run the tests? `ls -la coach-bot-repo/content/tests/*.webp`
- **Solution:** Run tests first: `cd coach-bot-repo && pytest src/content_generators/additional_content/stimulus_image/drawing_functions/tests/ -v`

**Problem:** Images not matching functions
- **Solution:** Regenerate patterns: `python3 retrieve_function_images.py --mode update-mapping`
- **Check:** Verify patterns in `function_filename_mapping.json`

**Problem:** Tests failing
- **Check:** Are repository dependencies installed?
- **Check:** Is `conftest.py` correctly configured?
- **Solution:** Check test output for specific errors

### Standards Sync Issues

**Problem:** Standards not syncing
- **Check:** Are database credentials in `.env`?
- **Check:** Can you connect to the database?
- **Solution:** Test connection: `python3 -c "from src.database import connect_to_db; connect_to_db()"`

**Problem:** Standards sync is slow
- **Note:** This is normal - the script processes all functions concurrently (10 workers)
- **Wait:** Let it complete - it will show progress updates

### General Issues

**Problem:** Import errors
- **Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

**Problem:** Streamlit app won't start
- **Check:** Is Python 3.8+ installed? `python3 --version`
- **Check:** Is Streamlit installed? `pip show streamlit`

**Problem:** Cache issues
- **Solution:** Clear Streamlit cache: Delete `.streamlit/cache/` directory
- **Solution:** Restart the Streamlit app

---

## Quick Reference

### Complete Setup Sequence

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
# Edit .env file with your credentials

# 3. Clone repository
./setup_repo_without_token.sh

# 4. Run tests to generate images
cd coach-bot-repo
pytest src/content_generators/additional_content/stimulus_image/drawing_functions/tests/ -v
cd ..

# 5. Start the app
streamlit run app_new.py

# 6. In the app, click "📦 Sync Repository & Functions"
# 7. In the app, click "📚 Sync Standards"
```

### Regular Updates

```bash
# Update repository
./sync_coach_bot.sh

# Regenerate images (if needed)
cd coach-bot-repo
pytest src/content_generators/additional_content/stimulus_image/drawing_functions/tests/ -v
cd ..

# Then sync in the app using the buttons
```

---

## Additional Resources

- **README.md**: General project information and features
- **RATE_LIMITING_GUIDE.md**: Information about AI search rate limiting
- **TODO.md**: Project roadmap and future improvements

---

## Need Help?

If you encounter issues not covered in this guide:

1. Check the troubleshooting section above
2. Review error messages carefully
3. Check that all prerequisites are met
4. Verify environment variables are set correctly
5. Open an issue on GitHub with:
   - Your operating system
   - Python version
   - Error messages
   - Steps to reproduce

---

**Happy exploring! 🎯**
