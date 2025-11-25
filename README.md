# 🌍 CharityChain — AI + Web3 Powered Transparent Charity Platform

**Empowering donors with proof of real-world impact**

CharityChain is a decentralized donation platform that ensures every rupee donated creates verifiable, measurable, and authentic humanitarian impact. Using AI-powered evidence verification + NFT-based Proof-of-Impact tokens, donors finally get transparency and confidence like never before.

---

## 🚀 Key Features

| Module | What it does |
|--------|-------------|
| 🤖 **AI Evidence Verification** | Detects fraud, GPS mismatch, tampering, duplication & false progress |
| 🔗 **Web3-Inspired Smart Funding** | Funds only released after evidence is verified |
| 🪪 **Proof-of-Impact NFTs** | Donors receive collectible NFTs when milestones are achieved |
| 📊 **Impact Analytics Dashboard** | Live charts showing funds raised, verification success rate & progress |
| 🗺 **Global Project Map** | Visualizes real NGO impact locations using geocoordinates |
| 🏅 **Badge & Gamification System** | Rewards donors & NGOs for achievement and transparency |
| 💾 **Offline-Ready DB** | TinyDB ensures the app works even without an active blockchain |

---

## 🛠️ Architecture Overview

```
Streamlit Frontend (UI)
       │
       ▼
CharityTracker Backend Engine
       │
       ├── DatabaseManager (TinyDB)
       ├── AI Verifier (Image Auth + GPS + Similarity + ML)
       ├── IPFS Handler (Demo mode)
       ├── Oracle Verification Logic
       ├── NFT + Badge System
       └── Analytics + Visualization Engine
```

---

## 🧪 Demo Data (Instant Setup)

The platform auto-loads demo projects on first launch, meaning judges immediately see:
- ✅ 5 global charity projects
- ✅ Realistic donation progress
- ✅ Pending milestones for evidence upload

You can reload sample data anytime through:

```
🚀 Reload Demo Data  (Sidebar Button)
```

---

## 💻 How to Run Locally

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit

```bash
streamlit run app_ui.py
```

### 3️⃣ Browser opens automatically at:

```
http://localhost:8501/
```

---

## 🔍 Folder Structure

```
CharityChain/
│
├── app_ui.py                # Streamlit frontend
├── charity_tracker.py       # Backend: DB + AI + NFTs + Analytics
├── requirements.txt
├── README.md
│
├── data/                    # DB folder (auto-filled at runtime)
│   ├── charity.db          # TinyDB database (auto-generated)
│   └── evidence/           # Uploaded evidence files
│
├── logs/                    # Application logs
├── models/                  # Optional model weights (YOLOv8)
└── build/                   # Smart contract artifacts
```

⚠️ **Do NOT commit your local DB file (TinyDB JSON). It is auto-generated.**

---

## 📌 Technology Stack

| Category | Tech |
|----------|------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **AI** | Image Authenticity, GPS EXIF, Similarity & Tamper Detection |
| **Storage** | TinyDB |
| **Blockchain-Inspired** | NFT simulation, milestone escrow, donor tokens |
| **Deployment** | Streamlit Cloud / Render |

---

## 🤖 AI Verification System

CharityChain uses a **6-algorithm verification engine** to ensure authenticity:

### 1. **Object Detection (YOLOv8)**
- Identifies relevant objects in evidence photos
- Ensures claimed activities are actually present

### 2. **GPS Validation**
- Extracts EXIF geolocation data
- Verifies location within 1000m radius of project site

### 3. **Tamper Detection (ELA)**
- Error Level Analysis detects image manipulation
- Flags suspicious editing or Photoshop artifacts

### 4. **Perceptual Hashing (pHash)**
- Compares before/after images
- Detects genuine change vs. fake progress

### 5. **SSIM (Structural Similarity)**
- Measures image similarity scores
- Prevents duplicate evidence submission

### 6. **EXIF Metadata Analysis**
- Validates camera information
- Checks timestamps and device authenticity

**Confidence Score**: AI generates a 0-100% confidence score. Projects require ≥65% to pass verification.

---

## 🎮 Gamification & Rewards

### 🪙 Impact Tokens
- Earn **100 tokens per ETH donated**
- Redeemable for platform benefits

### 🏅 Achievement Badges

| Badge | Requirement |
|-------|-------------|
| 💎 **Top Donor** | Make 5+ donations |
| 🚀 **Early Supporter** | Make your first donation |
| 🏆 **Impact Champion** | Donate 10+ ETH total |
| ⭐ **Trusted NGO** | Complete 10+ verified milestones |
| ✅ **100% Verified** | Earn 5+ Impact NFTs |

### 🎨 NFT Collection
Every verified donation mints a unique **Proof-of-Impact NFT** containing:
- Project details
- Donation amount
- Verification timestamp
- IPFS evidence link

---

## 👥 User Roles

### 🏢 NGO (Charity Organization)
1. Create charitable projects with funding goals
2. Submit photo/video evidence when milestones are reached
3. Receive funds automatically after verification

### 💰 Donor
1. Browse and filter verified projects
2. Donate to causes they trust
3. Earn Impact Tokens and NFTs on verification

### 🔍 Oracle (Verifier)
1. Review AI verification reports
2. Approve/reject evidence submissions
3. Trigger fund release or rejection

### 👤 Guest
- View public projects and analytics
- Explore platform features without making transactions

---

## 📊 Analytics Dashboard

Real-time insights include:
- 📈 Total funds raised across all projects
- ✅ Verification success rates
- 🗺️ Geographic distribution of projects
- 👥 Top donors and NGOs
- 📅 Project timeline visualization
- 🎯 Status distribution (Pending/Verified/Released)

---

## 🔐 Security Features

- ✅ **MIME Type Validation** - Only accepts JPG/PNG images
- ✅ **File Size Limits** - Max 10MB per upload
- ✅ **Rate Limiting** - 10 uploads per hour per user
- ✅ **GPS Verification** - Ensures evidence location authenticity
- ✅ **Tamper Detection** - ELA analysis flags manipulated images
- ✅ **Image Sanitization** - Automatic resizing and format validation

---

## 🌐 Environment Variables (Optional)

For production deployment with real blockchain:

```bash
# Blockchain Configuration
WEB3_PROVIDER=https://polygon-rpc.com
CHAIN_ID=137
ORACLE_PRIVATE_KEY=0x...

# IPFS Storage
WEB3_STORAGE_TOKEN=eyJ...
PINATA_API_KEY=...
PINATA_SECRET=...

# AI Features (Optional)
OPENAI_API_KEY=sk-...
```

**Demo Mode**: Leave empty or set to `DEMO_MODE` to run without blockchain.

---

## 🐛 Troubleshooting

### "No milestones found"
- Click **🚀 Reload Demo Data** in sidebar
- Check `data/charity.db` exists

### YOLO detection fails
```bash
pip install ultralytics
```
Model auto-downloads on first run.

### IPFS upload timeout
- Check internet connection
- Verify `WEB3_STORAGE_TOKEN` if using production mode
- Demo mode uses mock uploads

### Database locked error
- Restart application
- Delete `data/charity.db` (resets all data)

---

## 🏆 Why This Project Wins Hackathons

✔️ **Solves a real humanitarian problem**  
✔️ **Combines AI + Web3 in a meaningful way**  
✔️ **Demonstrates full end-to-end flow live in the demo**  
✔️ **Includes impact, gamification & transparency**  
✔️ **Takes under 3 minutes to pitch and wows judges visually**

---

## 🚧 Future Roadmap

- [ ] Multi-chain support (Ethereum, BSC, Avalanche)
- [ ] Mobile app (React Native)
- [ ] DAO governance for oracle decisions
- [ ] Integration with real NGO APIs (GiveIndia, GlobalGiving)
- [ ] Social media sharing of NFTs
- [ ] Advanced fraud detection ML models
- [ ] Multi-language support (Hindi, Spanish, French)
- [ ] SMS notifications for milestone updates

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 🧑‍💻 Contributors

| Name | Role |
|------|------|
| **Meenal Sinha** | Project Lead & Developer |

Want to join the team? Open an issue or PR!

---

## 📄 License

This project is released under the **MIT License** — free to use and modify.

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenCV** for image analysis
- **Ultralytics** for YOLOv8
- **Streamlit** for rapid UI development
- **Web3.Storage** for decentralized storage
- **Polygon** for scalable blockchain infrastructure
- **TinyDB** for lightweight database

---

## 📞 Support

- 🐛 **GitHub Issues**: [Create an issue](https://github.com/yourusername/charitychain/issues)
- 📧 **Email**: support@charitychain.org

---

**Built with ❤️ by the CharityChain Team**

**Version**: 2.0 | **Status**: Production Ready 🚀

⭐ **If you like this project, please give the repository a star!** ⭐
