# 🌍 CharityChain — AI-Powered Transparent Charity Platform

**Empowering donors with proof of real-world impact**

CharityChain is a donation platform that ensures every rupee donated creates verifiable, measurable, and authentic humanitarian impact. Using AI-powered evidence verification and digital Proof-of-Impact certificates, donors finally get transparency and confidence like never before.

---

## 🚀 Key Features

| Module | What it does |
|--------|-------------|
| 🤖 **AI Evidence Verification** | Detects fraud, tampering, and validates evidence authenticity |
| 🔗 **Smart Funding Logic** | Simulates escrow behavior - funds only released after verification |
| 🪪 **Proof-of-Impact Certificates** | Donors receive digital certificates when milestones are achieved |
| 📊 **Impact Analytics Dashboard** | Live charts showing funds raised, verification success rate & progress |
| 🗺 **Global Project Map** | Visualizes real NGO impact locations using geocoordinates |
| 🏅 **Badge & Gamification System** | Rewards donors & NGOs for achievement and transparency |
| 💾 **Lightweight Database** | TinyDB ensures fast local development and prototyping |

---

## 🛠️ Architecture Overview

```
Streamlit Frontend (UI)
       │
       ▼
CharityTracker Backend Engine
       │
       ├── DatabaseManager (TinyDB)
       ├── AI Verifier (Object Detection + Tamper Detection + Image Analysis)
       ├── Mock IPFS Handler (Demo mode with simulated CIDs)
       ├── Oracle Verification Logic
       ├── Certificate & Badge System
       └── Analytics + Visualization Engine
```

---

## 🧪 Demo Data (Instant Setup)

The platform auto-loads demo projects on first launch, meaning you immediately see:
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
├── charity_tracker.py       # Backend: DB + AI + Certificates + Analytics
├── requirements.txt
├── README.md
│
├── data/                    # DB folder (auto-filled at runtime)
│   ├── charity.db          # TinyDB database (auto-generated)
│   └── evidence/           # Uploaded evidence files
│
├── logs/                    # Application logs
└── models/                  # Optional model weights (YOLOv8)
```

⚠️ **Do NOT commit your local DB file (TinyDB JSON). It is auto-generated.**

---

## 📌 Technology Stack

| Category | Tech |
|----------|------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **AI** | YOLOv8 Object Detection, Image Analysis, Tamper Detection |
| **Storage** | TinyDB (local JSON database) |
| **Visualization** | Plotly, Altair, Folium |
| **Deployment** | Streamlit Cloud / Render |

---

## 🤖 AI Verification System

CharityChain uses multiple verification algorithms to ensure authenticity:

### 1. **Object Detection (YOLOv8)**
- Identifies relevant objects in evidence photos
- Ensures claimed activities are actually present
- Detects people, construction equipment, infrastructure, etc.

### 2. **Tamper Detection (Error Level Analysis)**
- Analyzes compression artifacts to detect image manipulation
- Flags suspicious editing or Photoshop artifacts
- Uses ELA (Error Level Analysis) algorithm

### 3. **Image Similarity Analysis**
- Compares before/after images using perceptual hashing (pHash)
- Detects genuine change vs. fake progress
- Prevents duplicate evidence submission

### 4. **Metadata Inspection**
- Extracts EXIF data from images
- Validates camera information and timestamps
- Checks for metadata tampering

### 5. **Location Mapping**
- Maps user-provided coordinates on interactive map
- Validates proximity to expected project location
- *(EXIF GPS extraction planned for future)*

**Confidence Score**: AI generates a 0-100% confidence score based on all checks. Projects require ≥65% to pass verification.

---

## 🎮 Gamification & Rewards

### 🪙 Impact Points System
- Earn **100 points per ETH donated**
- Gamified engagement to encourage contributions
- *(Blockchain token minting planned for future)*

### 🏅 Achievement Badges

| Badge | Requirement |
|-------|-------------|
| 💎 **Top Donor** | Make 5+ donations |
| 🚀 **Early Supporter** | Make your first donation |
| 🏆 **Impact Champion** | Donate 10+ ETH total |
| ⭐ **Trusted NGO** | Complete 10+ verified milestones |
| ✅ **100% Verified** | Earn 5+ Impact Certificates |

### 🎨 Certificate Collection
Every verified donation generates a unique **Proof-of-Impact Certificate** containing:
- Project details
- Donation amount
- Verification timestamp
- Evidence reference
- *(On-chain NFT minting planned for future)*

---

## 👥 User Roles

### 🏢 NGO (Charity Organization)
1. Create charitable projects with funding goals
2. Submit photo/video evidence when milestones are reached
3. Receive simulated fund release after verification

### 💰 Donor
1. Browse and filter verified projects
2. Donate to causes they trust
3. Earn Impact Points and Certificates on verification

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
- ✅ **Location Validation** - Verifies project coordinates
- ✅ **Tamper Detection** - ELA analysis flags manipulated images
- ✅ **Image Sanitization** - Automatic resizing and format validation

---

## 🌐 Environment Variables (Optional)

For future blockchain integration:

```bash
# Planned Blockchain Configuration
WEB3_PROVIDER=https://polygon-rpc.com
CHAIN_ID=137
ORACLE_PRIVATE_KEY=0x...

# Planned IPFS Storage
WEB3_STORAGE_TOKEN=eyJ...
PINATA_API_KEY=...
PINATA_SECRET=...

# AI Features (Optional)
OPENAI_API_KEY=sk-...
```

**Current Mode**: All features work in demo/simulation mode without requiring blockchain or IPFS credentials.

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

### Database locked error
- Restart application
- Delete `data/charity.db` (resets all data)

---

## 🏆 Why This Project Stands Out

✔️ **Solves a real humanitarian problem** - Addresses charity transparency crisis  
✔️ **Working AI verification** - Functional multi-algorithm evidence validation  
✔️ **Full demo flow** - Complete donor → NGO → oracle → verification cycle  
✔️ **Rich visualizations** - Interactive maps, charts, and dashboards  
✔️ **Clean architecture** - Separated UI and backend for maintainability  
✔️ **Gamification done right** - Badges and achievements drive engagement  

---

## 🚧 Future Roadmap

### Phase 1: Enhanced AI
- [ ] EXIF GPS extraction and validation
- [ ] SSIM (Structural Similarity) comparison
- [ ] Advanced ML fraud detection models
- [ ] Video evidence support

### Phase 2: Blockchain Integration
- [ ] Smart contract deployment (Ethereum/Polygon)
- [ ] On-chain NFT minting (ERC-721)
- [ ] Token rewards system (ERC-20)
- [ ] Decentralized oracle network

### Phase 3: Platform Expansion
- [ ] Real IPFS integration
- [ ] Mobile app (React Native)
- [ ] Integration with NGO APIs (GiveIndia, GlobalGiving)
- [ ] Multi-language support (Hindi, Spanish, French)
- [ ] SMS notifications for milestone updates
- [ ] DAO governance for oracle decisions

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

---

## 📄 License

This project is released under the **MIT License** — free to use and modify.

See [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenCV** for image analysis
- **Ultralytics** for YOLOv8
- **Streamlit** for rapid UI development
- **TinyDB** for lightweight database
- **Plotly & Altair** for beautiful visualizations
- **Folium** for interactive maps

---

## 📞 Support

- 🐛 **GitHub Issues**: [Create an issue](https://github.com/yourusername/charitychain/issues)
- 📧 **Email**: support@charitychain.org

---

**Built with ❤️ by the CharityChain Team**

**Version**: 2.0 | **Status**: Working Demo 🚀

⭐ **If you like this project, please give the repository a star!** ⭐
