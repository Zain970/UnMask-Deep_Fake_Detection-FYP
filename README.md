# 🎭 UnMask - Deep Fake Detection System

<div align="center">
  <img src="./webpage.png" alt="UnMask Web Interface" width="800"/>
  
  [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
  [![Documentation](https://img.shields.io/badge/docs-ready-blue.svg)](docs/)
</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🌟 Overview

UnMask is an advanced deep fake detection system that uses state-of-the-art AI models to identify manipulated media content. Our system provides a user-friendly interface for uploading and analyzing media files, with detailed reports on detection results.

## ✨ Features

### 🖼️ Media Analysis

- Deep fake detection for images and videos
- Multiple AI model support (Xception, EfficientNet, ResNet)
- Real-time processing and analysis
- Detailed detection reports with confidence scores

### 🔒 Security & Authentication

- Secure user authentication
- Role-based access control
- API key management
- Secure file upload handling

### 📊 Monitoring & Analytics

- Real-time performance monitoring
- Comprehensive error tracking
- System resource monitoring
- Detailed analytics dashboard

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PostgreSQL database
- 8GB RAM minimum
- 20GB free disk space

### Setup Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/BSM21/UNMASK.git
   cd UnMask
   ```

2. **Create virtual environment**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python scripts/init_db.py
   ```

## 💻 Usage

### Web Interface

1. Start the development server:
   ```bash
   python run.py
   ```
2. Access the web interface at `http://localhost:8000`

### API Integration

```python
import requests

# Upload media for analysis
response = requests.post(
    'http://localhost:8000/api/analyze',
    files={'media': open('test.jpg', 'rb')}
)
result = response.json()
```

## 📁 Project Structure

```
UnMask/
├── ai_model/           # AI model implementations
│   ├── models/        # Model architectures
│   └── preprocessing/ # Image preprocessing
├── front_end/         # Web interface
│   ├── static/       # CSS, JS, images
│   └── templates/    # HTML templates
├── scripts/          # Utility scripts
├── tests/           # Test suite
├── utils/           # Utility modules
└── docs/           # Documentation
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation as needed
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Project Link:** (https://github.com/Zain970/UnMask-Deep_Fake_Detection-FYP)
- **Email:** bilal.saeed.malik123@gmail.com or zainulabidin970946@gmail.com

## 🙏 Acknowledgments

- TensorFlow and Keras teams
- OpenCV community
- All contributors and maintainers
- Open source community

---

<div align="center">
  <sub>Built with ❤️ for a better future</sub>
</div>
