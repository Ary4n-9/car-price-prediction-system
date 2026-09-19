
import os
import html
import base64
import json
import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# Smart Car Price & Recommendation System
# ============================================================

MODEL_PATH = "car_price_predictor.pkl"
DATA_PATH = "car data.csv"
IMAGE_DIR = "car_images"
REFERENCE_YEAR = 2026

model = joblib.load(MODEL_PATH)
cars = pd.read_csv(DATA_PATH)

cars["Age"] = REFERENCE_YEAR - cars["Year"]
cars = cars.rename(
    columns={
        "Selling_Price": "Selling_Price(lacs)",
        "Present_Price": "Present_Price(lacs)",
        "Owner": "Past_Owners",
    }
)

recommendation_features = [
    "Selling_Price(lacs)",
    "Present_Price(lacs)",
    "Kms_Driven",
    "Past_Owners",
    "Age",
]


# ============================================================
# Recommendation Engine
# ============================================================

def recommend_cars(
    predicted_price,
    present_price,
    kms_driven,
    fuel_type,
    seller_type,
    transmission,
    past_owners,
    age,
    top_n=3,
):
    rec = cars.copy()

    numeric_input = np.array(
        [
            predicted_price,
            present_price,
            kms_driven,
            past_owners,
            age,
        ],
        dtype=float,
    )

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(rec[recommendation_features])
    scaled_input = scaler.transform([numeric_input])[0]

    numeric_distance = np.sqrt(
        np.sum((scaled_data - scaled_input) ** 2, axis=1)
    )

    categorical_penalty = (
        (rec["Fuel_Type"] != fuel_type).astype(float) * 0.60
        + (rec["Seller_Type"] != seller_type).astype(float) * 0.40
        + (rec["Transmission"] != transmission).astype(float) * 0.40
    )

    rec["match_distance"] = numeric_distance + categorical_penalty

    return (
        rec.sort_values("match_distance")
        .drop_duplicates(subset=["Car_Name"])
        .head(top_n)
        .copy()
    )


# ============================================================
# UI Helpers
# ============================================================

def money(value):
    return f"₹{value:.2f} Lakhs"


def safe_text(value):
    return html.escape(str(value))



def normalize_name(value):
    """Normalize dataset car names for image lookup."""
    return " ".join(str(value).strip().lower().split())


def get_car_image_html(car_name):
    """
    Load a local car image from car_images/ and embed it as base64.
    This works on Render because the image files are part of the repository.
    """
    normalized = normalize_name(car_name)

    # image_map.json contains only images that are represented in the
    # generated catalog. Names without a matching image use the fallback.
    map_path = os.path.join(IMAGE_DIR, "image_map.json")

    try:
        with open(map_path, "r", encoding="utf-8") as f:
            image_map = json.load(f)

        relative_path = image_map.get(normalized)

        if relative_path:
            image_path = relative_path
            if not os.path.isabs(image_path):
                image_path = os.path.join(".", image_path)

            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")

                return (
                    f'<img class="real-car-image" '
                    f'src="data:image/png;base64,{encoded}" '
                    f'alt="{safe_text(car_name)}">'
                )
    except Exception:
        pass

    # Fallback for dataset names not represented in the generated catalog.
    return """
    <div class="image-fallback">
        <span>🚗</span>
    </div>
    """


def car_card(rank, row):
    car_name = safe_text(row["Car_Name"])
    price = float(row["Selling_Price(lacs)"])
    fuel = safe_text(row["Fuel_Type"])
    transmission = safe_text(row["Transmission"])
    kms = int(row["Kms_Driven"])
    car_age = int(row["Age"])

    rank_class = {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "silver")
    car_image = get_car_image_html(row["Car_Name"])

    return f"""
    <div class="car-card {rank_class}">
        <div class="card-top">
            <span class="rank-badge">{rank}</span>
            <span class="heart">♡</span>
        </div>

        <div class="car-image">
            {car_image}
        </div>

        <div class="car-name">{car_name}</div>

        <div class="car-price">{money(price)}</div>

        <div class="spec-grid">
            <span>⛽ {fuel}</span>
            <span>⚙️ {transmission}</span>
            <span>🛣️ {kms:,} km</span>
            <span>📅 {car_age} years</span>
        </div>
    </div>
    """


def prediction_ui(
    present_price,
    kms_driven,
    fuel_type,
    seller_type,
    transmission,
    past_owners,
    age,
):
    try:
        if present_price is None or kms_driven is None or age is None:
            raise ValueError("Please enter all required car details.")

        input_df = pd.DataFrame(
            [{
                "Present_Price(lacs)": float(present_price),
                "Kms_Driven": float(kms_driven),
                "Fuel_Type": fuel_type,
                "Seller_Type": seller_type,
                "Transmission": transmission,
                "Past_Owners": int(past_owners),
                "Age": int(age),
            }]
        )

        predicted_price = max(0.0, float(model.predict(input_df)[0]))

        top = recommend_cars(
            predicted_price=predicted_price,
            present_price=float(present_price),
            kms_driven=float(kms_driven),
            fuel_type=fuel_type,
            seller_type=seller_type,
            transmission=transmission,
            past_owners=int(past_owners),
            age=int(age),
            top_n=3,
        )

        cards = "".join(
            car_card(rank, (_, row)[1])
            for rank, (_, row) in enumerate(top.iterrows(), start=1)
        )

        price_html = f"""
        <div class="price-panel">
            <div class="price-heading">
                <span class="result-icon">▥</span>
                <div>
                    <div class="eyebrow">ESTIMATED SELLING PRICE</div>
                    <div class="sub-heading">Based on your car details</div>
                </div>
            </div>

            <div class="big-price">{money(predicted_price)}</div>

            <div class="estimate-note">
                <span class="check">✓</span>
                Estimated price generated by the machine-learning model.
            </div>
        </div>
        """

        recommendation_html = f"""
        <div class="recommend-section">
            <div class="recommend-header">
                <div>
                    <div class="recommend-title">
                        <span class="star">★</span> Top 3 Recommended Cars
                    </div>
                    <div class="recommend-subtitle">
                        Closest matches from your dataset
                    </div>
                </div>
                <span class="dataset-pill">Best dataset matches</span>
            </div>

            <div class="cards-row">
                {cards}
            </div>
        </div>
        """

        return price_html, recommendation_html

    except Exception as e:
        return (
            f"""
            <div class="price-panel error-panel">
                <div class="eyebrow">PREDICTION ERROR</div>
                <div class="error-message">{safe_text(e)}</div>
            </div>
            """,
            "",
        )


def reset_ui():
    return (
        8.0,
        35000,
        "Petrol",
        "Dealer",
        "Manual",
        0,
        5,
        "",
        "",
    )


# ============================================================
# CSS
# ============================================================

CSS = """
:root {
    --navy: #0c2342;
    --blue: #1769ff;
    --blue2: #20a4ff;
    --light: #f4f8ff;
    --border: #d9e5f5;
    --text: #13294b;
    --muted: #63758e;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: #f5f9ff !important;
}

footer {
    display: none !important;
}

/* ---------- Hero ---------- */

.hero {
    min-height: 185px;
    padding: 25px 55px 20px;
    color: white;
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 86% 35%, rgba(47,156,255,.30), transparent 24%),
        linear-gradient(105deg, #092442 0%, #123b67 50%, #172d50 100%);
    border-radius: 0 0 24px 24px;
}

.hero:after {
    content: "";
    position: absolute;
    width: 450px;
    height: 180px;
    right: 50px;
    bottom: -80px;
    border-radius: 50%;
    background: rgba(60, 157, 255, .16);
    filter: blur(10px);
}

.hero-inner {
    position: relative;
    z-index: 2;
}

.brand-line {
    display: flex;
    align-items: center;
    gap: 15px;
}

.car-logo {
    font-size: 52px;
    line-height: 1;
}

.hero-title {
    margin: 0;
    font-size: 39px;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-title span {
    color: #26a8ff;
}

.hero-subtitle {
    margin: 7px 0 0 68px;
    font-size: 16px;
    color: #d9e9ff;
}

.feature-row {
    display: flex;
    gap: 10px;
    margin-top: 20px;
}

.feature {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 18px;
    border-right: 1px solid rgba(255,255,255,.15);
}

.feature-icon {
    width: 35px;
    height: 35px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: rgba(255,255,255,.10);
    font-size: 19px;
}

.feature strong {
    display: block;
    font-size: 14px;
}

.feature small {
    color: #b7cce6;
    font-size: 11px;
}

/* ---------- Main ---------- */

.main-wrap {
    padding: 22px 32px 15px;
}

.input-card,
.results-wrap {
    background: white;
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: 0 8px 28px rgba(24, 66, 112, .08);
}

.input-card {
    padding: 24px 24px 20px;
}

.section-heading {
    display: flex;
    gap: 13px;
    align-items: center;
    margin-bottom: 20px;
}

.section-icon {
    width: 45px;
    height: 45px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: linear-gradient(135deg, #277cff, #1ba7ff);
    color: white;
    font-size: 22px;
}

.section-heading h2 {
    margin: 0;
    color: var(--text);
    font-size: 23px;
}

.section-heading p {
    margin: 3px 0 0;
    color: var(--muted);
    font-size: 13px;
}

/* Gradio inputs */

.input-card label span {
    color: var(--text) !important;
    font-weight: 700 !important;
}

.input-card input,
.input-card select {
    border-radius: 10px !important;
    border: 1px solid #d5e1ef !important;
    min-height: 44px !important;
}

.input-card input:focus,
.input-card select:focus {
    border-color: #2886ff !important;
    box-shadow: 0 0 0 3px rgba(40,134,255,.10) !important;
}

.predict-btn {
    min-height: 52px !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    background: linear-gradient(100deg, #1e55ff, #12a7ff) !important;
    border: 0 !important;
    box-shadow: 0 8px 18px rgba(30, 100, 255, .20) !important;
}

.reset-btn {
    min-height: 52px !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
}

/* ---------- Results ---------- */

.results-wrap {
    padding: 20px;
}

.price-panel {
    background:
        radial-gradient(circle at 95% 30%, rgba(43,150,255,.18), transparent 25%),
        linear-gradient(135deg, #eef8ff, #e4f3ff);
    border-radius: 16px;
    padding: 20px 25px 19px;
    border: 1px solid #cce7ff;
}

.price-heading {
    display: flex;
    align-items: center;
    gap: 12px;
}

.result-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: #d9edff;
    color: #1769ff;
    display: grid;
    place-items: center;
    font-size: 25px;
    font-weight: 900;
}

.eyebrow {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: .5px;
    color: var(--text);
}

.sub-heading {
    color: var(--muted);
    font-size: 12px;
    margin-top: 2px;
}

.big-price {
    font-size: 45px;
    line-height: 1;
    font-weight: 900;
    color: #1555e8;
    margin: 15px 0;
    letter-spacing: -1.5px;
}

.estimate-note {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 9px 14px;
    border-radius: 10px;
    color: #176c3b;
    background: #dff8e9;
    font-size: 12px;
    font-weight: 600;
}

.check {
    width: 20px;
    height: 20px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    color: white;
    background: #1ea35c;
    font-weight: 900;
}

.recommend-section {
    margin-top: 17px;
}

.recommend-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 3px 12px;
}

.recommend-title {
    color: var(--text);
    font-size: 21px;
    font-weight: 850;
}

.star {
    color: #ffb800;
    margin-right: 5px;
}

.recommend-subtitle {
    margin: 2px 0 0 29px;
    color: var(--muted);
    font-size: 12px;
}

.dataset-pill {
    background: #edf5ff;
    border: 1px solid #d8e7fb;
    color: #365a83;
    border-radius: 20px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 700;
}

.cards-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 13px;
}

.car-card {
    min-width: 0;
    border-radius: 15px;
    border: 1px solid #dce6f2;
    background: linear-gradient(180deg, #ffffff, #f9fbff);
    padding: 12px;
    transition: transform .2s ease, box-shadow .2s ease;
}

.car-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(28, 71, 117, .12);
}

.car-card.gold {
    border-color: #f5d273;
    background: linear-gradient(180deg, #fffdf7, #ffffff);
}

.car-card.silver {
    border-color: #dce5ef;
}

.car-card.bronze {
    border-color: #efd2c4;
    background: linear-gradient(180deg, #fffaf7, #ffffff);
}

.card-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.rank-badge {
    width: 31px;
    height: 31px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    background: #e7effb;
    color: #234b7a;
    font-weight: 900;
    font-size: 13px;
}

.gold .rank-badge {
    background: #ffc83d;
    color: #6c4800;
}

.bronze .rank-badge {
    background: #d9915d;
    color: white;
}

.heart {
    color: #e03b42;
    font-size: 24px;
}

.car-image {
    height: 108px;
    display: grid;
    place-items: center;
    margin-top: 2px;
    overflow: hidden;
    border-radius: 11px;
    background: linear-gradient(180deg, #f7fbff, #edf4fb);
}

.real-car-image {
    width: 100%;
    height: 108px;
    object-fit: contain;
    display: block;
}

.image-fallback {
    width: 100%;
    height: 108px;
    display: grid;
    place-items: center;
    font-size: 52px;
    background: linear-gradient(180deg, #f7fbff, #edf4fb);
}

.car-name {
    text-align: center;
    color: var(--text);
    font-size: 16px;
    font-weight: 850;
    margin-top: 2px;
    text-transform: capitalize;
}

.car-price {
    text-align: center;
    margin: 8px 0 10px;
    padding: 7px;
    border-radius: 16px;
    color: #1555e8;
    background: #e4efff;
    font-size: 15px;
    font-weight: 900;
}

.spec-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 5px;
    color: #53677f;
    font-size: 11px;
    padding: 2px 2px 3px;
}

.spec-grid span {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.error-panel {
    background: #fff2f2;
    border-color: #ffcaca;
}

.error-message {
    margin-top: 10px;
    color: #a12626;
    font-weight: 700;
}

/* ---------- Footer ---------- */

.footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 11px 38px 15px;
    color: #61738c;
    font-size: 11px;
}

.footer strong {
    color: #203b5e;
}

@media (max-width: 900px) {
    .hero {
        padding: 22px 25px;
    }

    .hero-title {
        font-size: 30px;
    }

    .hero-subtitle {
        margin-left: 0;
    }

    .feature-row {
        flex-wrap: wrap;
    }

    .main-wrap {
        padding: 15px;
    }

    .cards-row {
        grid-template-columns: 1fr;
    }

    .footer {
        padding: 10px 18px;
    }
}
"""


# ============================================================
# Gradio App
# ============================================================

with gr.Blocks(
    title="Smart Car Price Predictor",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-inner">
            <div class="brand-line">
                <div class="car-logo">🚙</div>
                <div>
                    <h1 class="hero-title">
                        Smart Car <span>Price Predictor</span>
                    </h1>
                </div>
            </div>

            <div class="hero-subtitle">
                Predict the selling price of a used car and get matching car recommendations.
            </div>

            <div class="feature-row">
                <div class="feature">
                    <div class="feature-icon">🎯</div>
                    <div>
                        <strong>AI Powered</strong>
                        <small>Machine Learning Predictions</small>
                    </div>
                </div>

                <div class="feature">
                    <div class="feature-icon">⚡</div>
                    <div>
                        <strong>Instant Results</strong>
                        <small>Price & car suggestions</small>
                    </div>
                </div>

                <div class="feature">
                    <div class="feature-icon">🛡️</div>
                    <div>
                        <strong>Smart Insights</strong>
                        <small>Data-driven car matching</small>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)

    with gr.Row(elem_classes="main-wrap"):

        # LEFT: INPUT
        with gr.Column(scale=4, elem_classes="input-card"):

            gr.HTML("""
            <div class="section-heading">
                <div class="section-icon">⚙</div>
                <div>
                    <h2>Enter Car Details</h2>
                    <p>Fill in the details to predict the price and find similar cars.</p>
                </div>
            </div>
            """)

            with gr.Row():
                present_price = gr.Number(
                    label="Present Price (Lakhs)",
                    value=8.5,
                    minimum=0.1,
                    placeholder="e.g. 8.5",
                )
                kms_driven = gr.Number(
                    label="Kilometres Driven",
                    value=35000,
                    minimum=0,
                    precision=0,
                    placeholder="e.g. 35000",
                )

            with gr.Row():
                fuel_type = gr.Dropdown(
                    choices=["Petrol", "Diesel", "CNG"],
                    value="Petrol",
                    label="Fuel Type",
                )
                seller_type = gr.Dropdown(
                    choices=["Dealer", "Individual"],
                    value="Dealer",
                    label="Seller Type",
                )

            with gr.Row():
                transmission = gr.Dropdown(
                    choices=["Manual", "Automatic"],
                    value="Manual",
                    label="Transmission",
                )
                past_owners = gr.Dropdown(
                    choices=[0, 1, 2, 3],
                    value=0,
                    label="Previous Owners",
                )

            age = gr.Slider(
                minimum=0,
                maximum=20,
                value=5,
                step=1,
                label="Car Age (Years)",
                info="Move the slider to set the vehicle age.",
            )

            with gr.Row():
                predict_btn = gr.Button(
                    "✨ Predict Price & Find Cars",
                    variant="primary",
                    elem_classes="predict-btn",
                )
                reset_btn = gr.Button(
                    "↻ Reset",
                    elem_classes="reset-btn",
                )

        # RIGHT: RESULTS
        with gr.Column(scale=7, elem_classes="results-wrap"):

            price_output = gr.HTML("""
            <div class="price-panel">
                <div class="price-heading">
                    <span class="result-icon">▥</span>
                    <div>
                        <div class="eyebrow">ESTIMATED SELLING PRICE</div>
                        <div class="sub-heading">Your prediction will appear here</div>
                    </div>
                </div>
                <div class="big-price">₹-- Lakhs</div>
                <div class="estimate-note">
                    <span class="check">✓</span>
                    Enter details and click the prediction button.
                </div>
            </div>
            """)

            recommendation_output = gr.HTML("""
            <div class="recommend-section">
                <div class="recommend-header">
                    <div>
                        <div class="recommend-title">
                            <span class="star">★</span> Top 3 Recommended Cars
                        </div>
                        <div class="recommend-subtitle">
                            Matching cars will appear here after prediction.
                        </div>
                    </div>
                    <span class="dataset-pill">Dataset matches</span>
                </div>

                <div class="cards-row">
                    <div class="car-card">
                        <div class="car-image">🚗</div>
                        <div class="car-name">Waiting for prediction</div>
                    </div>
                </div>
            </div>
            """)

    gr.HTML("""
    <div class="footer">
        <div>
            🚙 <strong>Smart Car Price Predictor</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp; Built with ❤️ using Machine Learning
        </div>
        <div>
            <strong>Better Data. Smarter Decisions. A Better Drive.</strong>
        </div>
    </div>
    """)

    predict_btn.click(
        fn=prediction_ui,
        inputs=[
            present_price,
            kms_driven,
            fuel_type,
            seller_type,
            transmission,
            past_owners,
            age,
        ],
        outputs=[price_output, recommendation_output],
        show_progress="minimal",
    )

    reset_btn.click(
        fn=reset_ui,
        outputs=[
            present_price,
            kms_driven,
            fuel_type,
            seller_type,
            transmission,
            past_owners,
            age,
            price_output,
            recommendation_output,
        ],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
