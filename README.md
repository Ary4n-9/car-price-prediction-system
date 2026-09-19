# 🚗 Smart Car Price & Recommendation System

A Gradio machine-learning application that predicts used-car selling price and recommends the Top 3 matching car names.

## Features

- Wide, responsive dashboard interface
- ML-based selling price prediction
- Top 3 matching car recommendations
- Local car images from `car_images/`
- Render-compatible Gradio server
- No external image hosting required

## Project files

```text
app.py
car_price_predictor.pkl
car data.csv
requirements.txt
.gitignore
car_images/
    image_map.json
    car_catalog.png
    *.png
```

The generated catalog supplied for the project contains passenger-car images. Only dataset names represented in that catalog receive a matching image; other dataset names use a clean fallback icon.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
python app.py
```
