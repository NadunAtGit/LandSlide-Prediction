# Vision-Based Landslide Forecasting

This project investigates landslide forecasting using mapped landslide boundaries and Sentinel-2 satellite imagery.

The project uses the Cyclone Ditwah 2025 landslide boundary dataset and retrieves Sentinel-2 Level-2A satellite imagery programmatically through the Microsoft Planetary Computer STAC API.

## Setup

### 1. Clone the Repository

```bash
git clone 
cd "LandSlide Prediction"

python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
