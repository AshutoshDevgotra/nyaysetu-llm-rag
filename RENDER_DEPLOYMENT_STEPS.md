# Render Deployment - Step-by-Step Guide

## ✅ Step 1: Code Pushed to GitHub
Your code is now on GitHub with all deployment files!

---

## 🚀 Step 2: Create Render Web Service

### Option A: Using Blueprint (Recommended - Easiest)
1. Go to **[Render Dashboard](https://dashboard.render.com)**
2. Click **"New +"** → **"Blueprint"**
3. Click **"Connect GitHub"** (if not already connected)
4. Find and select your **`nyaysetu-llm-rag`** repository
5. Render will auto-detect `render.yaml` and show the service configuration
6. Click **"Apply"**

### Option B: Manual Web Service
1. Go to **[Render Dashboard](https://dashboard.render.com)**
2. Click **"New +"** → **"Web Service"**
3. Connect your **`nyaysetu-llm-rag`** repository
4. Configure:
   - **Name**: `nyaysetu-rag-backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free

---

## 🔐 Step 3: Add Environment Variables

After creating the service, go to **Environment** tab and add:

| Key | Value | Where to Get |
|-----|-------|--------------|
| `GEMINI_API_KEY` | Your Gemini API key | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| `PINECONE_API_KEY` | Your Pinecone API key | [Pinecone Console](https://app.pinecone.io/) |
| `LLM_PROVIDER` | `gemini` | (already in render.yaml) |
| `LLM_MODEL` | `gemini-2.0-flash` | (already in render.yaml) |
| `LLM_TEMPERATURE` | `0.2` | (already in render.yaml) |
| `LLM_MAX_OUTPUT_TOKENS` | `256` | (already in render.yaml) |

> **Note**: Only `GEMINI_API_KEY` and `PINECONE_API_KEY` need to be added manually. The others are already configured in `render.yaml`.

---

## 🎯 Step 4: Deploy

1. Click **"Create Web Service"** or **"Manual Deploy"**
2. Wait for build to complete (2-3 minutes)
3. Your backend will be live at: `https://nyaysetu-rag-backend.onrender.com`

---

## ✅ Step 5: Test Your Deployment

### Test Health Endpoint
Visit: `https://your-app.onrender.com/health`

Expected response:
```json
{
  "status": "healthy",
  "llm_type": "gemini",
  "llm_model": "gemini-2.0-flash"
}
```

### Test Debug Endpoint
Visit: `https://your-app.onrender.com/debug_info`

Expected response:
```json
{
  "llm_provider": "gemini",
  "llm_model": "gemini-2.0-flash",
  "vectorstore_type": "Pinecone"
}
```

### Test Ask Endpoint
```bash
curl -X POST https://your-app.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is contract law?"}'
```

---

## 📱 Step 6: Update Frontend

After deployment, update your frontend to use the production backend:

**For Vercel/Production:**
Add environment variable in Vercel dashboard:
```
NEXT_PUBLIC_BACKEND_URL=https://your-app.onrender.com
```

**For local development:**
Keep using:
```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8082
```

---

## ⚠️ Important Notes

**Free Tier**: 
- Spins down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds
- Perfect for testing and development

**Paid Tier** ($7/month):
- No spin-down delays
- Always ready to respond
- Recommended for production

---

## 🐛 Troubleshooting

**Build fails?**
- Check Render logs for errors
- Verify all dependencies in `requirements.txt`
- Ensure Python 3.10+ is used

**Runtime errors?**
- Verify environment variables are set
- Check Pinecone index `nyayadwaar-gemini` exists
- Verify Gemini API key is valid

**Connection issues?**
- Check CORS settings allow your frontend domain
- Verify health endpoint returns 200 OK
- Ensure APIs are accessible from Render

---

## 📋 Quick Checklist

- [ ] Go to Render Dashboard
- [ ] Create new Blueprint or Web Service
- [ ] Connect GitHub repository
- [ ] Add GEMINI_API_KEY environment variable
- [ ] Add PINECONE_API_KEY environment variable
- [ ] Deploy service
- [ ] Test /health endpoint
- [ ] Test /debug_info endpoint
- [ ] Test /ask endpoint
- [ ] Update frontend with production URL

**You're ready to deploy! 🚀**
