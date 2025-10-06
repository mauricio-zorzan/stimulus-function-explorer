# Stimulus Function Explorer

A simple app to explore and manage stimulus functions from your GitHub repository.

## Features

- 📋 Browse all stimulus functions from your repo
- 🔍 View function details, code, and tests
- 🎨 Store and view generated images in S3
- 🔄 Real-time sync with your GitHub repository

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-new-repo>
cd stimulus-function-explorer
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```
# GitHub Configuration
GITHUB_TOKEN=ghp_your_token_here
GITHUB_REPO_OWNER=your_username
GITHUB_REPO_NAME=coach-bot-external-content-generators

# AWS S3 Configuration (optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET_NAME=stimulus-function-images
AWS_REGION=us-east-1
```

### 3. Run the App

```bash
streamlit run app_new.py
```

## Setting Up GitHub Token

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with `repo` permissions
3. Copy the token to your `.env` file

## Setting Up S3 (Optional)

1. Create an AWS account
2. Create an S3 bucket for storing images
3. Create IAM user with S3 permissions
4. Add credentials to `.env` file

## Usage

1. **Function List**: Browse all available stimulus functions
2. **Function Details**: View code, tests, and documentation
3. **Images**: View generated sample images (requires S3)

## Next Steps

- Add image generation from test cases
- Add AI-powered image analysis
- Add function enhancement recommendations
- Add automated testing

## Development

To add new features:

1. Create new modules in `src/`
2. Update `app_new.py` to include new functionality
3. Test with your GitHub repository

## Troubleshooting

- **GitHub API rate limits**: Use a personal access token
- **Missing functions**: Check if your repo structure matches expected paths
