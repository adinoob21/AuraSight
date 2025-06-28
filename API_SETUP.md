# API Key Setup Guide

## Security Notice
⚠️ **Never commit API keys to version control!** The `.env` file is already in `.gitignore` to prevent accidental commits.

## Required API Keys

### 1. Cohere API Key (Required)
The application uses Cohere for natural language processing and generating responses.

**How to get it:**
1. Go to [Cohere.ai](https://cohere.ai/)
2. Sign up for a free account
3. Navigate to your API keys section
4. Copy your API key

**Setup:**
1. Create a file named `.env` in the project root
2. Add your API key:
   ```
   COHERE_API_KEY=your_actual_api_key_here
   ```

### 2. AssemblyAI API Key (Optional)
Used for advanced speech-to-text functionality.

**How to get it:**
1. Go to [AssemblyAI](https://www.assemblyai.com/)
2. Sign up for an account
3. Get your API key from the dashboard

**Setup:**
Add to your `.env` file:
```
STT_API_KEY=your_assemblyai_api_key_here
```

## Example .env File

Create a file named `.env` in the project root with this content:

```env
# AuraSight Environment Variables

# Cohere API Key for natural language processing
# Get your free API key from: https://cohere.ai/
COHERE_API_KEY=your_cohere_api_key_here

# AssemblyAI API Key for speech-to-text (optional)
# Get your API key from: https://www.assemblyai.com/
STT_API_KEY=your_assemblyai_api_key_here

# Add other API keys as needed
# OPENAI_API_KEY=your_openai_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here
```

## Verification

After setting up your `.env` file, run the application:

```bash
python main.py
```

You should see:
- ✅ Cohere API client initialized successfully (if Cohere key is set)
- ⚠️ Warning messages for missing API keys

## Troubleshooting

### "COHERE_API_KEY not found" Error
1. Make sure you created a `.env` file (not `.env.txt`)
2. Check that the file is in the project root directory
3. Verify the API key format: `COHERE_API_KEY=your_key_here`
4. Restart the application after creating the `.env` file

### API Key Not Working
1. Verify your API key is correct
2. Check if you have sufficient credits/quota
3. Ensure your internet connection is working
4. Try regenerating your API key

### Security Best Practices
1. **Never share your API keys** - they are like passwords
2. **Use different keys for development and production**
3. **Rotate keys regularly** for better security
4. **Monitor your API usage** to avoid unexpected charges
5. **Keep your `.env` file secure** and never commit it to git

## Free Tier Limits

### Cohere
- Free tier includes generous limits
- No credit card required for basic usage
- Check [Cohere pricing](https://cohere.ai/pricing) for current limits

### AssemblyAI
- Free tier available with usage limits
- Check [AssemblyAI pricing](https://www.assemblyai.com/pricing) for current limits

## Support

If you're having trouble with API setup:
1. Check the official documentation for each service
2. Verify your API keys are active and have sufficient credits
3. Test your keys with the service's API testing tools
4. Contact the service's support if needed 