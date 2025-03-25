# GitHub-Railway Integration Guide

This guide provides detailed instructions for setting up continuous deployment from GitHub to Railway for the Enhanced Sports Prediction System.

## Prerequisites

- GitHub account
- Railway account
- Git installed on your local machine

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in to your account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Name your repository (e.g., "sports-prediction-system")
4. Add a description (optional)
5. Choose "Public" or "Private" visibility
6. Check "Initialize this repository with a README"
7. Click "Create repository"

## Step 2: Clone the Repository to Your Local Machine

```bash
git clone https://github.com/your-username/sports-prediction-system.git
cd sports-prediction-system
```

## Step 3: Add Project Files to the Repository

1. Copy all files from the enhanced sports prediction system to your local repository:

```bash
# Assuming the enhanced_sports_prediction directory is in the same parent directory
cp -r ../enhanced_sports_prediction/* .
```

2. Make sure the following Railway configuration files are included:
   - Procfile
   - railway.json
   - requirements.txt
   - database/railway_db_config.py
   - web_interface/railway_app.py

3. Create a .gitignore file to exclude unnecessary files:

```bash
# Create .gitignore file
cat > .gitignore << EOL
__pycache__/
*.py[cod]
*$py.class
*.so
.env
.venv
env/
venv/
ENV/
.idea/
.vscode/
*.log
*.sqlite
EOL
```

4. Commit and push the changes to GitHub:

```bash
git add .
git commit -m "Initial commit of Enhanced Sports Prediction System"
git push origin main
```

## Step 4: Connect Railway to Your GitHub Repository

1. Log in to [Railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. If this is your first time, you'll need to authorize Railway to access your GitHub account
5. Choose your "sports-prediction-system" repository from the list
6. Railway will automatically detect the Procfile and railway.json configuration
7. Click "Deploy" to start the initial deployment

## Step 5: Set Up the PostgreSQL Database

1. In your Railway project, click "New Service"
2. Select "Database" → "PostgreSQL"
3. Wait for the database to be provisioned
4. Railway will automatically link this database to your application by setting environment variables

## Step 6: Configure Environment Variables

1. In your Railway project, go to the "Variables" tab
2. Add all your API keys and other environment variables:
   - SPORTRADAR_API_KEY
   - ESPN_API_KEY
   - STATS_PERFORM_API_KEY
   - WEATHER_API_KEY
   - SECRET_KEY (for Flask)
3. Click "Add" to save the variables

## Step 7: Initialize the Database Schema

After the first deployment completes:

1. Go to the "Deployments" tab in your Railway project
2. Click on the latest deployment
3. Go to the "Shell" tab
4. Run the database initialization command:

```bash
python -c "from database.railway_db_config import get_db_connection, initialize_database; initialize_database(get_db_connection())"
```

## Step 8: Generate a Public Domain

1. In your Railway project, go to the "Settings" tab
2. Click "Generate Domain"
3. Railway will provide a public URL for your application

## Step 9: Set Up Continuous Deployment

Railway automatically sets up continuous deployment. Whenever you push changes to your GitHub repository:

1. Make changes to your local repository:

```bash
# Make changes to files
git add .
git commit -m "Description of changes"
git push origin main
```

2. Railway will automatically detect the push and start a new deployment
3. You can monitor the deployment progress in the Railway dashboard

## Step 10: Managing Deployments

1. **View Deployment Logs**:
   - Go to the "Deployments" tab in your Railway project
   - Click on a deployment to view its logs

2. **Rollback to Previous Deployment**:
   - Go to the "Deployments" tab
   - Find the deployment you want to roll back to
   - Click the three dots menu and select "Redeploy"

3. **Configure Deployment Settings**:
   - Go to the "Settings" tab
   - You can configure auto-deployment settings, environment variables, and more

## Troubleshooting

### Common Issues

1. **Deployment Fails**:
   - Check the deployment logs for error messages
   - Verify that all required dependencies are in requirements.txt
   - Ensure the Procfile is correctly configured

2. **Database Connection Issues**:
   - Verify that the PostgreSQL service is properly linked to your application
   - Check that the environment variables are correctly set

3. **Application Crashes After Deployment**:
   - Check the logs for error messages
   - Verify that all environment variables are correctly set
   - Ensure the database schema has been initialized

### Getting Help

If you encounter issues, you can:
1. Check the Railway documentation: https://docs.railway.app/
2. Join the Railway Discord community: https://discord.gg/railway
3. Contact Railway support through their website

## Benefits of GitHub-Railway Integration

- **Continuous Deployment**: Changes are automatically deployed when you push to GitHub
- **Version Control**: All your code changes are tracked
- **Collaboration**: Easier for team members to contribute
- **Rollbacks**: You can easily revert to previous versions if needed
- **Branch Deployments**: You can set up Railway to deploy from specific branches for staging environments
