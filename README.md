# 🚗 Car Price Prediction API

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-009688.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A professional **FastAPI-based REST API** that predicts car prices using a **Gradient Boosting** machine learning model. Simply provide a car brand, and the API returns an intelligent price prediction along with the closest matching vehicle from a real user database.

## ✨ Features

- 🎯 **Accurate Price Predictions** - Uses Gradient Boosting ML model trained on real car specifications
- 🔍 **Smart Matching** - Finds the closest matching car listing from user database
- 🚀 **High Performance** - Built with FastAPI for blazing-fast responses
- 📊 **Rich Data** - Trained on 200+ car models with detailed specifications
- 🐳 **Docker Ready** - Containerized deployment with single command
- 📖 **Interactive Documentation** - Auto-generated API docs with Swagger UI
- 🔧 **CLI Tool** - Command-line interface for quick predictions
- 🌐 **CORS Enabled** - Ready for web application integration

## 🛠️ Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.9+ | Core programming language |
| **FastAPI** | 0.95.2 | High-performance web framework |
| **scikit-learn** | 1.2.2 | Machine learning model |
| **Pandas** | 2.0.0 | Data manipulation |
| **Uvicorn** | 0.22.0 | ASGI server |
| **Docker** | Latest | Containerization |

## 📁 Project Structure

```
Car-Info/
├── 📄 README.md                    # Project documentation
├── 📄 .gitignore                   # Git ignore patterns
├── 📄 Dockerfile                   # Docker configuration
├── 📄 requirements.txt             # Python dependencies
├── 📄 runtime.txt                  # Python runtime version
│
├── 📂 src/                         # Source code
│   ├── __init__.py                 # Package initializer
│   ├── main.py                     # FastAPI application
│   └── predict_cli.py              # CLI prediction tool
│
├── 📂 models/                      # Machine learning models
│   └── gradient_boosting_model_v2.joblib
│
└── 📂 data/                        # Datasets
    ├── model.csv                   # Car specifications (205 models)
    └── user.csv                    # User car listings
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Car-Info.git
   cd Car-Info
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

**Start the API server:**
```bash
uvicorn src.main:app --reload
```

The API will be available at: `http://localhost:8000`

**Access Interactive Documentation:**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📖 Usage

### API Endpoint

#### Predict Car Price

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "brand": "Maruti"
}
```

**Response:**
```json
{
  "predicted_price": 750000,
  "match": {
    "fuel_type": 3,
    "engine_displacement": 1197,
    "no_cylinder": 4,
    "seating_capacity": 5,
    "transmission_type": 0,
    "fuel_tank_capacity": 37,
    "body_type": 2,
    "max_torque_nm": 113,
    "max_torque_rpm": 4400
  }
}
```

**Status Codes:**
- `200 OK` - Successful prediction
- `404 Not Found` - Brand not found in dataset
- `422 Unprocessable Entity` - Invalid request format

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"brand\": \"Toyota\"}"
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"brand": "Hyundai"}
)

data = response.json()
print(f"Predicted Price: ₹{data['predicted_price']:,}")
```

### CLI Tool

Run the command-line interface for quick predictions:

```bash
python src/predict_cli.py
```

**Example interaction:**
```
Loading model and data...

🚗 Enter car brand: Tata

💰 Predicted Price: ₹1,084,000 INR

✅ Found matching car at the predicted price:

╒═══════════╤═════════════════════╤══════════════╤══════════════════════╤══════════════════════╕
│ fuel_type │ engine_displacement │ no_cylinder  │ seating_capacity     │ transmission_type    │
╞═══════════╪═════════════════════╪══════════════╪══════════════════════╪══════════════════════╡
│     1     │        1497         │      4       │          5           │          0           │
╘═══════════╧═════════════════════╧══════════════╧══════════════════════╧══════════════════════╛
```

## 🐳 Docker Deployment

### Build the Docker image

```bash
docker build -t car-price-api .
```

### Run the container

```bash
docker run -d -p 8080:8080 --name car-api car-price-api
```

Access the API at: `http://localhost:8080`

### Docker Compose (Optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8080:8080"
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## 🤖 Model Information

### Machine Learning Approach

- **Algorithm**: Gradient Boosting Regressor
- **Training Data**: 205 car models with 11 features
- **Features Used**:
  - Fuel type
  - Engine displacement
  - Number of cylinders
  - Seating capacity
  - Transmission type
  - Fuel tank capacity
  - Body type
  - Max torque (Nm & RPM)
  - Max power (BHP & RPM)

### Model Performance

The model uses a sophisticated matching algorithm:
1. Predicts price based on brand specifications
2. Calculates dynamic tolerance (50% of predicted price, minimum ₹20,000)
3. Finds closest matching car in user database
4. Returns prediction with actual listing details

## 🔧 Development

### Project Setup for Development

```bash
# Clone repository
git clone https://github.com/yourusername/Car-Info.git
cd Car-Info

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run in development mode
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Code Style

This project follows:
- **PEP 8** style guidelines
- **Type hints** for better code clarity
- **Docstrings** for all functions and classes

## 📊 Supported Brands

The API supports predictions for the following brands:

<details>
<summary>Click to expand brand list</summary>

- Maruti
- Mahindra
- Toyota
- Hyundai
- Tata
- Kia
- Honda
- Land Rover
- MG
- Citroen
- Nissan
- Renault
- Lamborghini
- Volkswagen
- Skoda
- Mercedes-Benz
- Volvo
- Jeep
- BMW
- Audi
- Force
- Ferrari
- Rolls-Royce
- Jaguar
- Porsche
- Lexus
- Bentley
- Maserati
- Aston Martin
- McLaren
- Mini
- BYD
- Datsun
- Isuzu
- Bajaj
- Strom Motors
- Compass

</details>

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/AmazingFeature`
3. **Commit your changes**: `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch**: `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 coding standards
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📝 API Response Codes

| Code | Description |
|------|-------------|
| 200  | Success - Price predicted successfully |
| 404  | Not Found - Brand not in dataset |
| 422  | Validation Error - Invalid input format |
| 500  | Server Error - Internal server error |

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Solution**: Ensure you're running commands from the project root directory

**Issue**: `FileNotFoundError: model file not found`
- **Solution**: Verify that `models/gradient_boosting_model_v2.joblib` exists

**Issue**: Port already in use
- **Solution**: Change the port: `uvicorn src.main:app --port 8001`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML powered by [scikit-learn](https://scikit-learn.org/)
- Data processing with [Pandas](https://pandas.pydata.org/)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by [Your Name]

</div>
