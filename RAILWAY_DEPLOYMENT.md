# Railway Deployment Instructions

This file contains instructions for deploying the Enhanced Sports Prediction System on Railway.com.

## Files for Railway Deployment

1. **Procfile**: Tells Railway how to run your application
2. **railway.json**: Configuration file for Railway
3. **requirements.txt**: Lists all Python dependencies
4. **database/railway_db_config.py**: Database configuration for Railway
5. **web_interface/railway_app.py**: Flask application configured for Railway

## Deployment Steps

### 1. Set Up Railway Account

1. Sign up for an account at [Railway.app](https://railway.app/)
2. Install the Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```
3. Login to Railway:
   ```bash
   railway login
   ```

### 2. Deploy the Database

1. Create a new PostgreSQL database on Railway:
   ```bash
   railway add
   ```
   Select "PostgreSQL" when prompted

2. Note the database connection details from the Railway dashboard

### 3. Deploy the Application

1. Initialize your project with Railway:
   ```bash
   cd enhanced_sports_prediction
   railway init
   ```

2. Link to your existing project (where you created the PostgreSQL database):
   ```bash
   railway link
   ```

3. Deploy your application:
   ```bash
   railway up
   ```

4. Set up environment variables in the Railway dashboard:
   - Go to your project in the Railway dashboard
   - Click on "Variables"
   - Add your API keys:
     - SPORTRADAR_API_KEY
     - ESPN_API_KEY
     - STATS_PERFORM_API_KEY
     - WEATHER_API_KEY
     - SECRET_KEY (for Flask)

5. Generate a public domain:
   ```bash
   railway domain
   ```

### 4. Initialize the Database

After deployment, you need to initialize the database schema:

1. Connect to your Railway project:
   ```bash
   railway connect
   ```

2. Run the database initialization script:
   ```bash
   python -c "from database.railway_db_config import get_db_connection, initialize_database; initialize_database(get_db_connection())"
   ```

### 5. Verify Deployment

1. Visit your Railway-generated domain to access the web interface
2. Check the logs in the Railway dashboard to ensure everything is running correctly

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Verify that the PostgreSQL service is properly linked to your application
   - Check that the environment variables are correctly set

2. **Application Not Starting**:
   - Check the logs in the Railway dashboard
   - Ensure all dependencies are listed in requirements.txt

3. **Missing Environment Variables**:
   - Verify all required environment variables are set in the Railway dashboard

### Getting Help

If you encounter issues, you can:
1. Check the Railway documentation: https://docs.railway.app/
2. Join the Railway Discord community: https://discord.gg/railway
3. Contact Railway support through their website
