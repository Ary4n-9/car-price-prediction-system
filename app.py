import os
import html
import json
import base64

import joblib
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# SMART CAR PRICE & RECOMMENDATION SYSTEM
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
# HELPERS
# ============================================================

def safe_text(value):
    return html.escape(str(value))


def money(value):
    return f"₹{float(value):.2f} Lakhs"


def normalize_name(value):
    return " ".join(str(value).strip().lower().split())


def get_car_image(car_name):
    """Return a local image as a base64 HTML image."""
    map_path = os.path.join(IMAGE_DIR, "image_map.json")

    try:
        with open(map_path, "r", encoding="utf-8") as f:
            image_map = json.load(f)

        relative_path = image_map.get(normalize_name(car_name))

        if relative_path:
            image_path = relative_path
            if not os.path.isabs(image_path):
                image_path = os.path.join(".", image_path)

            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")

                ext = os.path.splitext(image_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"

                return (
                    f'<img class="car-photo" '
                    f'src="data:{mime};base64,{encoded}" '
                    f'alt="{safe_text(car_name)}">'
                )
    except Exception:
        pass

    return '<div class="car-photo-fallback">🚗</div>'


# ============================================================
# RECOMMENDATION ENGINE
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

    scaled_data = scaler.fit_transform(
        rec[recommendation_features]
    )

    scaled_input = scaler.transform(
        [numeric_input]
    )[0]

    numeric_distance = np.sqrt(
        np.sum(
            (scaled_data - scaled_input) ** 2,
            axis=1,
        )
    )

    categorical_penalty = (
        (rec["Fuel_Type"] != fuel_type).astype(float) * 0.60
        + (rec["Seller_Type"] != seller_type).astype(float) * 0.40
        + (rec["Transmission"] != transmission).astype(float) * 0.40
    )

    rec["match_distance"] = (
        numeric_distance + categorical_penalty
    )

    return (
        rec.sort_values("match_distance")
        .drop_duplicates(subset=["Car_Name"])
        .head(top_n)
        .copy()
    )


# ============================================================
# RECOMMENDATION CARD
# ============================================================

def make_car_card(rank, row):
    car_name = safe_text(row["Car_Name"])
    price = float(row["Selling_Price(lacs)"])
    fuel = safe_text(row["Fuel_Type"])
    transmission = safe_text(row["Transmission"])
    kms = int(row["Kms_Driven"])
    age = int(row["Age"])

    rank_class = {
        1: "rank-one",
        2: "rank-two",
        3: "rank-three",
    }.get(rank, "rank-two")

    image = get_car_image(row["Car_Name"])

    return f"""
    <div class="car-card {rank_class}">

        <div class="card-header">
            <span class="rank-badge">{rank}</span>
            <span class="heart">♡</span>
        </div>

        <div class="car-image-area">
            {image}
        </div>

        <div class="car-name">
            {car_name}
        </div>

        <div class="car-price">
            {money(price)}
        </div>

        <div class="car-specs">
            <div>
                <span class="spec-icon">⛽</span>
                <span>{fuel}</span>
            </div>

            <div>
                <span class="spec-icon">⚙</span>
                <span>{transmission}</span>
            </div>

            <div>
                <span class="spec-icon">▱</span>
                <span>{kms:,} km</span>
            </div>

            <div>
                <span class="spec-icon">▣</span>
                <span>{age} years</span>
            </div>
        </div>

    </div>
    """


# ============================================================
# PREDICTION
# ============================================================

def predict_car(
    present_price,
    kms_driven,
    fuel_type,
    seller_type,
    transmission,
    past_owners,
    age,
):
    try:
        if present_price is None:
            raise ValueError("Please enter the present price.")

        if kms_driven is None:
            raise ValueError("Please enter kilometres driven.")

        if age is None:
            raise ValueError("Please enter the car age.")

        input_df = pd.DataFrame(
            [
                {
                    "Present_Price(lacs)": float(present_price),
                    "Kms_Driven": float(kms_driven),
                    "Fuel_Type": fuel_type,
                    "Seller_Type": seller_type,
                    "Transmission": transmission,
                    "Past_Owners": int(past_owners),
                    "Age": int(age),
                }
            ]
        )

        predicted_price = max(
            0.0,
            float(model.predict(input_df)[0]),
        )

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

        cards = ""

        for rank, (_, row) in enumerate(
            top.iterrows(),
            start=1,
        ):
            cards += make_car_card(rank, row)

        price_html = f"""
        <div class="price-card">

            <div class="price-title-row">
                <div class="price-icon">▥</div>

                <div>
                    <div class="small-title">
                        ESTIMATED SELLING PRICE
                    </div>

                    <div class="small-subtitle">
                        Based on your car details
                    </div>
                </div>
            </div>

            <div class="big-price">
                {money(predicted_price)}
            </div>

            <div class="estimate-message">
                <span class="check-circle">✓</span>
                Estimated price generated by the machine-learning model.
            </div>

        </div>
        """

        recommendation_html = f"""
        <div class="recommendation-panel">

            <div class="recommendation-heading">

                <div>
                    <div class="recommendation-title">
                        <span class="star">★</span>
                        Top 3 Recommended Cars
                    </div>

                    <div class="recommendation-subtitle">
                        Closest matches based on your car details
                    </div>
                </div>

                <div class="match-badge">
                    Best dataset matches
                </div>

            </div>

            <div class="cars-grid">
                {cards}
            </div>

        </div>
        """

        return price_html, recommendation_html

    except Exception as e:

        return (
            f"""
            <div class="price-card error-card">
                <div class="small-title">
                    PREDICTION ERROR
                </div>

                <div class="error-text">
                    {safe_text(e)}
                </div>
            </div>
            """,
            "",
        )


def reset_form():
    return (
        8.5,
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
# CUSTOM CSS — MODERNIZED
# ============================================================

CSS = r"""

/* -----------------------------------------------------------
   TOKENS
----------------------------------------------------------- */

:root {
    --bg: #f5f7fb;
    --surface: #ffffff;
    --surface-2: #f8fafd;
    --border: #e6ebf3;
    --border-soft: #eef2f8;
    --ink: #10192b;
    --ink-soft: #64748b;
    --ink-faint: #94a3b8;

    --accent: #6d5ef7;
    --accent-2: #22c1dc;
    --accent-ink: #4c3fd6;
    --accent-soft: #f0edfe;

    --success: #17a367;
    --success-soft: #e4f9ee;
    --danger: #e5484d;
    --danger-soft: #fdeceb;

    --gold: #f2b705;
    --bronze: #e08e45;

    --radius-lg: 22px;
    --radius-md: 16px;
    --radius-sm: 12px;

    --shadow-sm: 0 1px 2px rgba(16, 25, 43, .04);
    --shadow-md: 0 10px 30px -12px rgba(16, 25, 43, .12);
    --shadow-lg: 0 20px 50px -18px rgba(76, 63, 214, .28);
}


/* -----------------------------------------------------------
   GLOBAL
----------------------------------------------------------- */

* {
    box-sizing: border-box;
}

body {
    background: var(--bg) !important;
}

.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family:
        "Inter",
        ui-sans-serif,
        system-ui,
        -apple-system,
        "Segoe UI",
        sans-serif !important;
}

footer {
    display: none !important;
}

::selection {
    background: var(--accent-soft);
    color: var(--accent-ink);
}


/* -----------------------------------------------------------
   HERO
----------------------------------------------------------- */

.hero {
    position: relative;
    overflow: hidden;

    margin: 18px 18px 0;
    padding: 40px 46px 34px;
    min-height: 190px;

    color: white;

    background:
        radial-gradient(circle at 15% 15%, rgba(255,255,255,.16), transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(34,193,220,.35), transparent 45%),
        linear-gradient(120deg, #241c5e 0%, #4c3fd6 48%, #6d5ef7 100%);

    border-radius: 28px;
    box-shadow: var(--shadow-lg);
}

.hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        radial-gradient(circle, rgba(255,255,255,.5) 1px, transparent 1px);
    background-size: 22px 22px;
    opacity: .05;
    pointer-events: none;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;

    margin-bottom: 14px;
    padding: 6px 13px;

    color: #e7e4ff;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 999px;

    font-size: 11px;
    font-weight: 700;
    letter-spacing: .5px;
    text-transform: uppercase;
    backdrop-filter: blur(6px);
}

.hero-title {
    margin: 0;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.3px;
    line-height: 1.08;
}

.hero-title span {
    background: linear-gradient(90deg, #9ff0ff, #c9c2ff);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.hero-subtitle {
    margin-top: 10px;
    max-width: 560px;

    color: #dcd9fb;
    font-size: 15.5px;
    line-height: 1.5;
}

.feature-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 26px;
}

.feature {
    display: flex;
    align-items: center;
    gap: 11px;

    padding: 10px 16px 10px 10px;

    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.16);
    border-radius: 14px;
    backdrop-filter: blur(6px);
}

.feature-icon {
    width: 34px;
    height: 34px;
    flex-shrink: 0;

    display: grid;
    place-items: center;

    border-radius: 10px;
    background: rgba(255,255,255,.14);

    font-size: 16px;
}

.feature strong {
    display: block;
    font-size: 13px;
    font-weight: 700;
}

.feature small {
    display: block;
    margin-top: 1px;
    color: #c9c5f4;
    font-size: 11px;
}


/* -----------------------------------------------------------
   MAIN LAYOUT
----------------------------------------------------------- */

.main-row {
    padding: 26px 18px 10px;
    gap: 22px !important;
    align-items: stretch !important;
}

.input-panel,
.results-panel {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-md) !important;
}

.input-panel {
    padding: 26px !important;
}

.results-panel {
    padding: 22px !important;
}


/* -----------------------------------------------------------
   INPUT HEADER
----------------------------------------------------------- */

.input-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 24px;
}

.input-header-icon {
    width: 46px;
    height: 46px;
    flex-shrink: 0;

    display: grid;
    place-items: center;

    border-radius: 14px;

    color: white;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));

    font-size: 20px;
    box-shadow: 0 8px 18px -6px rgba(109, 94, 247, .55);
}

.input-header h2 {
    margin: 0;
    color: var(--ink);
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -.3px;
}

.input-header p {
    margin: 3px 0 0;
    color: var(--ink-soft);
    font-size: 13px;
}


/* -----------------------------------------------------------
   FIELD CARDS
----------------------------------------------------------- */

.field-box {
    margin-bottom: 14px;
    padding: 13px 14px 14px;

    background: var(--surface-2);

    border: 1px solid var(--border-soft);
    border-radius: var(--radius-sm);
    transition: border-color .15s ease, background .15s ease;
}

.field-box:hover {
    border-color: #d9deee;
}

.field-label {
    display: block;
    margin-bottom: 7px;

    color: #334155;
    font-size: 12.5px;
    font-weight: 700;
    letter-spacing: -.1px;
}

.field-label span {
    color: var(--accent-ink);
}


/* -----------------------------------------------------------
   REMOVE GRADIO LABEL STYLING
----------------------------------------------------------- */

.clean-input label {
    display: none !important;
}

.clean-input > label {
    display: none !important;
}

.clean-input input,
.clean-input select,
.clean-input textarea {
    min-height: 44px !important;

    color: var(--ink) !important;
    background: var(--surface) !important;

    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;

    font-size: 14px !important;
    box-shadow: none !important;
    transition: border-color .15s ease, box-shadow .15s ease !important;
}

.clean-input input:hover,
.clean-input select:hover {
    border-color: #c7cee0 !important;
}

.clean-input input:focus,
.clean-input select:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--accent-soft) !important;
}


/* -----------------------------------------------------------
   AGE
----------------------------------------------------------- */

.age-box {
    margin-top: 2px;
    margin-bottom: 20px;

    padding: 16px;

    background:
        linear-gradient(135deg, var(--accent-soft), #eef9fb);

    border: 1px solid #e3ddfb;
    border-radius: var(--radius-sm);
}

.age-description {
    margin: -2px 0 10px;

    color: var(--ink-soft);
    font-size: 12px;
}


/* -----------------------------------------------------------
   BUTTONS
----------------------------------------------------------- */

.button-row {
    gap: 12px !important;
    margin-top: 6px;
}

.predict-button {
    min-height: 52px !important;

    border: none !important;
    border-radius: 13px !important;

    color: white !important;

    background:
        linear-gradient(100deg, var(--accent), var(--accent-2)) !important;

    font-size: 15px !important;
    font-weight: 750 !important;
    letter-spacing: -.1px !important;

    box-shadow: 0 12px 24px -8px rgba(109, 94, 247, .55) !important;
    transition: transform .15s ease, box-shadow .15s ease !important;
}

.predict-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 30px -8px rgba(109, 94, 247, .6) !important;
}

.predict-button:active {
    transform: translateY(0);
}

.reset-button {
    min-height: 52px !important;

    border: 1.5px solid var(--border) !important;
    border-radius: 13px !important;

    color: #475569 !important;
    background: var(--surface) !important;

    font-size: 15px !important;
    font-weight: 700 !important;
    transition: background .15s ease, border-color .15s ease !important;
}

.reset-button:hover {
    background: var(--surface-2) !important;
    border-color: #c7cee0 !important;
}


/* -----------------------------------------------------------
   PRICE RESULT
----------------------------------------------------------- */

.price-card {
    position: relative;
    overflow: hidden;

    padding: 26px 28px;

    border: 1px solid #e2ddfb;
    border-radius: var(--radius-md);

    background:
        radial-gradient(circle at 95% 15%, rgba(34,193,220,.14), transparent 40%),
        linear-gradient(135deg, #f6f4ff, #eef7ff);
}

.price-title-row {
    display: flex;
    align-items: center;
    gap: 13px;
}

.price-icon {
    width: 46px;
    height: 46px;

    display: grid;
    place-items: center;

    color: white;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));

    border-radius: 13px;

    font-size: 21px;
    font-weight: 900;
    box-shadow: 0 8px 18px -6px rgba(109, 94, 247, .5);
}

.small-title {
    color: #1e293b;
    font-size: 12.5px;
    font-weight: 800;
    letter-spacing: .5px;
}

.small-subtitle {
    margin-top: 3px;
    color: var(--ink-soft);
    font-size: 12px;
}

.big-price {
    margin: 18px 0 15px;

    background: linear-gradient(100deg, var(--accent-ink), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;

    font-size: 44px;
    line-height: 1;
    font-weight: 900;
    letter-spacing: -1.6px;
}

.estimate-message {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 9px 14px;

    color: var(--success);
    background: var(--success-soft);

    border-radius: 10px;

    font-size: 12px;
    font-weight: 650;
}

.check-circle {
    width: 20px;
    height: 20px;

    display: grid;
    place-items: center;

    color: white;
    background: var(--success);

    border-radius: 50%;

    font-weight: 900;
}


/* -----------------------------------------------------------
   RECOMMENDATIONS
----------------------------------------------------------- */

.recommendation-panel {
    margin-top: 20px;
}

.recommendation-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;

    margin-bottom: 15px;
}

.recommendation-title {
    display: flex;
    align-items: center;
    gap: 8px;

    color: #0f172a;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -.3px;
}

.star {
    color: var(--gold);
}

.recommendation-subtitle {
    margin-top: 3px;
    color: var(--ink-soft);
    font-size: 12px;
}

.match-badge {
    padding: 8px 14px;

    color: var(--accent-ink);
    background: var(--accent-soft);

    border: 1px solid #e3ddfb;
    border-radius: 999px;

    font-size: 11px;
    font-weight: 700;
}

.cars-grid {
    display: grid;

    grid-template-columns:
        repeat(3, minmax(0, 1fr));

    gap: 14px;
}


/* -----------------------------------------------------------
   CAR CARDS
----------------------------------------------------------- */

.car-card {
    min-width: 0;

    padding: 13px;

    border: 1px solid var(--border);
    border-radius: var(--radius-md);

    background: var(--surface);

    box-shadow: var(--shadow-sm);

    transition:
        transform .18s ease,
        box-shadow .18s ease,
        border-color .18s ease;
}

.car-card:hover {
    transform: translateY(-4px);
    border-color: #d9deee;
    box-shadow: var(--shadow-md);
}

.rank-one {
    border-color: #f5dfa0;
    background: linear-gradient(180deg, #fffbf0, #ffffff);
}

.rank-three {
    border-color: #f2d5bd;
    background: linear-gradient(180deg, #fff8f2, #ffffff);
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.rank-badge {
    width: 30px;
    height: 30px;

    display: grid;
    place-items: center;

    border-radius: 50%;

    color: var(--accent-ink);
    background: var(--accent-soft);

    font-size: 12.5px;
    font-weight: 900;
}

.rank-one .rank-badge {
    color: #8a5a00;
    background: linear-gradient(135deg, #ffe08a, var(--gold));
}

.rank-three .rank-badge {
    color: white;
    background: linear-gradient(135deg, var(--bronze), #c96a2a);
}

.heart {
    color: var(--danger);
    font-size: 21px;
    line-height: 1;
    transition: transform .12s ease;
}

.heart:hover {
    transform: scale(1.15);
}

.car-image-area {
    width: 100%;
    height: 112px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-top: 4px;

    overflow: hidden;

    border-radius: 11px;
    background: var(--surface-2);
}

.car-photo {
    display: block;

    width: 100%;
    height: 112px;

    object-fit: contain;

    background: var(--surface-2);
}

.car-photo-fallback {
    width: 100%;
    height: 112px;

    display: grid;
    place-items: center;

    color: #a3b1c6;
    background: var(--surface-2);

    font-size: 46px;
}

.car-name {
    min-height: 22px;

    margin-top: 8px;

    text-align: center;

    color: var(--ink);

    font-size: 14.5px;
    font-weight: 750;
    letter-spacing: -.2px;

    text-transform: capitalize;
}

.car-price {
    margin-top: 9px;
    padding: 8px;

    text-align: center;

    color: var(--accent-ink);
    background: var(--accent-soft);

    border-radius: 18px;

    font-size: 14.5px;
    font-weight: 800;
}

.car-specs {
    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 7px 5px;

    margin-top: 11px;
    padding: 0 2px 1px;

    color: var(--ink-soft);

    font-size: 10.5px;
}

.car-specs > div {
    display: flex;
    align-items: center;
    gap: 4px;

    min-width: 0;

    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.spec-icon {
    flex-shrink: 0;
}


/* -----------------------------------------------------------
   ERROR
----------------------------------------------------------- */

.error-card {
    background: var(--danger-soft);
    border-color: #f7c6c7;
}

.error-text {
    margin-top: 10px;
    color: #b3282d;
    font-weight: 700;
}


/* -----------------------------------------------------------
   FOOTER
----------------------------------------------------------- */

.footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;

    margin: 6px 18px 22px;
    padding: 16px 24px;

    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);

    color: var(--ink-soft);
    font-size: 11.5px;
}

.footer strong {
    color: #1e293b;
}


/* -----------------------------------------------------------
   RESPONSIVE
----------------------------------------------------------- */

@media (max-width: 1050px) {

    .main-row {
        padding: 20px 14px;
    }

    .cars-grid {
        grid-template-columns: 1fr;
    }

    .feature-row {
        gap: 10px;
    }
}

@media (max-width: 800px) {

    .hero {
        margin: 10px 10px 0;
        padding: 26px 22px;
    }

    .hero-title {
        font-size: 30px;
    }

    .feature-row {
        flex-direction: column;
        align-items: stretch;
    }

    .main-row {
        padding: 14px 10px;
    }

    .footer {
        flex-direction: column;
        align-items: flex-start;
        margin: 6px 10px 18px;
    }
}
"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    title="Smart Car Price Predictor",
    theme=gr.themes.Soft(),
    css=CSS,
) as demo:

    # HERO
    gr.HTML(
        """
        <div class="hero">

            <div class="hero-eyebrow">✦ Machine Learning · Used Car Pricing</div>

            <h1 class="hero-title">
                Smart Car <span>Price Predictor</span>
            </h1>

            <div class="hero-subtitle">
                Predict the fair selling price of a used car in seconds, and discover the closest matching listings from our dataset.
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
                    <div class="feature-icon">🛡</div>
                    <div>
                        <strong>Smart Insights</strong>
                        <small>Data-driven car matching</small>
                    </div>
                </div>

            </div>

        </div>
        """
    )


    # MAIN
    with gr.Row(
        elem_classes="main-row",
    ):

        # ====================================================
        # INPUT PANEL
        # ====================================================

        with gr.Column(
            scale=4,
            elem_classes="input-panel",
        ):

            gr.HTML(
                """
                <div class="input-header">

                    <div class="input-header-icon">
                        ⚙
                    </div>

                    <div>
                        <h2>Enter Car Details</h2>
                        <p>
                            Fill in the details to predict the price
                            and find similar cars.
                        </p>
                    </div>

                </div>
                """
            )


            # Row 1
            with gr.Row(equal_height=True):

                with gr.Column(
                    elem_classes="field-box",
                ):

                    gr.HTML(
                        '<div class="field-label">Present Price <span>(Lakhs)</span></div>'
                    )

                    present_price = gr.Number(
                        value=8.5,
                        minimum=0.1,
                        precision=2,
                        show_label=False,
                        container=False,
                        elem_classes="clean-input",
                    )


                with gr.Column(
                    elem_classes="field-box",
                ):

                    gr.HTML(
                        '<div class="field-label">Kilometres Driven</div>'
                    )

                    kms_driven = gr.Number(
                        value=35000,
                        minimum=0,
                        precision=0,
                        show_label=False,
                        container=False,
                        elem_classes="clean-input",
                    )


            # Row 2
            with gr.Row(equal_height=True):

                with gr.Column(
                    elem_classes="field-box",
                ):

                    gr.HTML(
                        '<div class="field-label">Fuel Type</div>'
                    )

                    fuel_type = gr.Dropdown(
                        choices=[
                            "Petrol",
                            "Diesel",
                            "CNG",
                        ],
                        value="Petrol",
                        show_label=False,
                        container=False,
                        elem_classes="clean-input",
                    )


                with gr.Column(
                    elem_classes="field-box",
                ):

                    gr.HTML(
                        '<div class="field-label">Seller Type</div>'
                    )

                    seller_type = gr.Dropdown(
                        choices=[
                            "Dealer",
                            "Individual",
                        ],
                        value="Dealer",
                        show_label=False,
                        container=False,
                        elem_classes="clean-input",
                    )


            # Row 3
            with gr.Row(equal_height=True):

                with gr.Column(
                    elem_classes="field-box",
                ):

                    gr.HTML(
                        '<div class="field-label">Transmission</div>'
                    )

                    transmission = gr.Dropdown(
                        choices=[
                            "Manual",
                            "Automatic",
                        ],
                        value="Manual",
                        show_label=False,
                        container=False,
                        elem_classes="clean-input",
                    )


                with gr.Column(
                    elem_classes="field-box",
                ):

                    gr.HTML(
                        '<div class="field-label">Previous Owners</div>'
                    )

                    past_owners = gr.Dropdown(
                        choices=[
                            0,
                            1,
                            2,
                            3,
                        ],
                        value=0,
                        show_label=False,
                        container=False,
                        elem_classes="clean-input",
                    )


            # AGE
            with gr.Column(
                elem_classes="age-box",
            ):

                gr.HTML(
                    '<div class="field-label">Car Age (Years)</div>'
                )

                gr.HTML(
                    '<div class="age-description">Enter the vehicle age in years.</div>'
                )

                age = gr.Number(
                    value=5,
                    minimum=0,
                    maximum=30,
                    precision=0,
                    show_label=False,
                    container=False,
                    elem_classes="clean-input",
                )


            # BUTTONS
            with gr.Row(
                elem_classes="button-row",
            ):

                predict_btn = gr.Button(
                    "✨ Predict Price & Find Cars",
                    variant="primary",
                    elem_classes="predict-button",
                )

                reset_btn = gr.Button(
                    "↻ Reset",
                    elem_classes="reset-button",
                )


        # ====================================================
        # RESULTS PANEL
        # ====================================================

        with gr.Column(
            scale=7,
            elem_classes="results-panel",
        ):

            price_output = gr.HTML(
                """
                <div class="price-card">

                    <div class="price-title-row">

                        <div class="price-icon">
                            ▥
                        </div>

                        <div>
                            <div class="small-title">
                                ESTIMATED SELLING PRICE
                            </div>

                            <div class="small-subtitle">
                                Based on your car details
                            </div>
                        </div>

                    </div>

                    <div class="big-price">
                        ₹-- Lakhs
                    </div>

                    <div class="estimate-message">
                        <span class="check-circle">✓</span>
                        Enter details and click Predict Price & Find Cars.
                    </div>

                </div>
                """
            )


            recommendation_output = gr.HTML(
                """
                <div class="recommendation-panel">

                    <div class="recommendation-heading">

                        <div>
                            <div class="recommendation-title">
                                <span class="star">★</span>
                                Top 3 Recommended Cars
                            </div>

                            <div class="recommendation-subtitle">
                                Closest matches based on your car details
                            </div>
                        </div>

                        <div class="match-badge">
                            Best dataset matches
                        </div>

                    </div>

                    <div class="cars-grid">

                        <div class="car-card">

                            <div class="car-image-area">
                                <div class="car-photo-fallback">
                                    🚗
                                </div>
                            </div>

                            <div class="car-name">
                                Waiting for prediction
                            </div>

                        </div>

                    </div>

                </div>
                """
            )


    # FOOTER
    gr.HTML(
        """
        <div class="footer">

            <div>
                🚙
                <strong>Smart Car Price Predictor</strong>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Built with ❤️ using Machine Learning
            </div>

            <div>
                <strong>
                    Better Data. Smarter Decisions. A Better Drive.
                </strong>
            </div>

        </div>
        """
    )


    # ========================================================
    # EVENTS
    # ========================================================

    predict_btn.click(
        fn=predict_car,
        inputs=[
            present_price,
            kms_driven,
            fuel_type,
            seller_type,
            transmission,
            past_owners,
            age,
        ],
        outputs=[
            price_output,
            recommendation_output,
        ],
        show_progress="minimal",
    )


    reset_btn.click(
        fn=reset_form,
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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860,
        )
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
