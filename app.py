
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
# CUSTOM CSS
# ============================================================

CSS = r"""

/* -----------------------------------------------------------
   GLOBAL
----------------------------------------------------------- */

* {
    box-sizing: border-box;
}

body {
    background: #f3f7fc !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    background: #f3f7fc !important;
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}

footer {
    display: none !important;
}


/* -----------------------------------------------------------
   HERO
----------------------------------------------------------- */

.hero {
    margin: 0;
    padding: 30px 55px 27px;
    min-height: 190px;

    color: white;

    background:
        radial-gradient(
            circle at 87% 25%,
            rgba(62, 164, 255, 0.30),
            transparent 25%
        ),
        linear-gradient(
            110deg,
            #0b2748 0%,
            #123d68 52%,
            #0d3157 100%
        );

    border-radius: 0 0 24px 24px;
}

.hero-title {
    margin: 0;
    font-size: 40px;
    font-weight: 850;
    letter-spacing: -1px;
}

.hero-title span {
    color: #31a9ff;
}

.hero-subtitle {
    margin-top: 8px;
    color: #dbeaff;
    font-size: 16px;
}

.feature-row {
    display: flex;
    gap: 28px;
    margin-top: 23px;
}

.feature {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-right: 28px;
    border-right: 1px solid rgba(255,255,255,.14);
}

.feature:last-child {
    border-right: none;
}

.feature-icon {
    width: 37px;
    height: 37px;

    display: grid;
    place-items: center;

    border-radius: 50%;
    background: rgba(255,255,255,.10);

    font-size: 18px;
}

.feature strong {
    display: block;
    font-size: 14px;
}

.feature small {
    display: block;
    margin-top: 2px;
    color: #bcd0e7;
    font-size: 11px;
}


/* -----------------------------------------------------------
   MAIN LAYOUT
----------------------------------------------------------- */

.main-row {
    padding: 25px 32px 10px;
    gap: 22px !important;
    align-items: stretch !important;
}

.input-panel,
.results-panel {
    background: #ffffff !important;
    border: 1px solid #dce7f3 !important;
    border-radius: 18px !important;

    box-shadow:
        0 8px 28px rgba(27, 65, 105, .07) !important;
}

.input-panel {
    padding: 25px !important;
}

.results-panel {
    padding: 20px !important;
}


/* -----------------------------------------------------------
   INPUT HEADER
----------------------------------------------------------- */

.input-header {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 23px;
}

.input-header-icon {
    width: 45px;
    height: 45px;

    display: grid;
    place-items: center;

    border-radius: 12px;

    color: white;
    background: linear-gradient(
        135deg,
        #2476f5,
        #16a4f7
    );

    font-size: 21px;
}

.input-header h2 {
    margin: 0;
    color: #122c4f;
    font-size: 22px;
    font-weight: 800;
}

.input-header p {
    margin: 3px 0 0;
    color: #71839a;
    font-size: 13px;
}


/* -----------------------------------------------------------
   FIELD CARDS
----------------------------------------------------------- */

.field-box {
    margin-bottom: 17px;
    padding: 14px 15px 15px;

    background: #f7faff;

    border: 1px solid #e2eaf4;
    border-radius: 13px;
}

.field-label {
    display: block;
    margin-bottom: 8px;

    color: #17365d;
    font-size: 13px;
    font-weight: 750;
}

.field-label span {
    color: #1b72ed;
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
    min-height: 46px !important;

    color: #17304f !important;
    background: #ffffff !important;

    border: 1px solid #cfdae8 !important;
    border-radius: 10px !important;

    box-shadow: none !important;
}

.clean-input input:hover,
.clean-input select:hover {
    border-color: #9ebce0 !important;
}

.clean-input input:focus,
.clean-input select:focus {
    border-color: #2c83f6 !important;

    box-shadow:
        0 0 0 3px rgba(44,131,246,.10) !important;
}


/* -----------------------------------------------------------
   AGE
----------------------------------------------------------- */

.age-box {
    margin-top: 2px;
    margin-bottom: 18px;

    padding: 15px;

    background: #f7faff;

    border: 1px solid #e2eaf4;
    border-radius: 13px;
}

.age-description {
    margin: -3px 0 10px;

    color: #71839a;
    font-size: 12px;
}


/* -----------------------------------------------------------
   BUTTONS
----------------------------------------------------------- */

.button-row {
    gap: 12px !important;
    margin-top: 4px;
}

.predict-button {
    min-height: 52px !important;

    border: none !important;
    border-radius: 12px !important;

    color: white !important;

    background:
        linear-gradient(
            100deg,
            #1760f5,
            #12a3f4
        ) !important;

    font-size: 15px !important;
    font-weight: 800 !important;

    box-shadow:
        0 8px 18px rgba(28,104,241,.20) !important;

    transition: transform .15s ease,
                box-shadow .15s ease !important;
}

.predict-button:hover {
    transform: translateY(-1px);
    box-shadow:
        0 11px 23px rgba(28,104,241,.27) !important;
}

.reset-button {
    min-height: 52px !important;

    border: 1px solid #d3dfec !important;
    border-radius: 12px !important;

    color: #29415e !important;
    background: #f6f9fd !important;

    font-size: 15px !important;
    font-weight: 750 !important;
}

.reset-button:hover {
    background: #edf4fb !important;
}


/* -----------------------------------------------------------
   PRICE RESULT
----------------------------------------------------------- */

.price-card {
    padding: 23px 25px;

    border: 1px solid #cfe5fb;
    border-radius: 16px;

    background:
        radial-gradient(
            circle at 92% 30%,
            rgba(57,153,245,.16),
            transparent 27%
        ),
        linear-gradient(
            135deg,
            #eef8ff,
            #e5f3ff
        );
}

.price-title-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.price-icon {
    width: 45px;
    height: 45px;

    display: grid;
    place-items: center;

    color: #1768e9;
    background: #dceeff;

    border-radius: 12px;

    font-size: 23px;
    font-weight: 900;
}

.small-title {
    color: #153153;
    font-size: 13px;
    font-weight: 850;
    letter-spacing: .3px;
}

.small-subtitle {
    margin-top: 3px;
    color: #657a94;
    font-size: 12px;
}

.big-price {
    margin: 15px 0 14px;

    color: #1557e8;

    font-size: 43px;
    line-height: 1;
    font-weight: 900;

    letter-spacing: -1.5px;
}

.estimate-message {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    padding: 9px 13px;

    color: #17683b;
    background: #def7e8;

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
    background: #20a25c;

    border-radius: 50%;

    font-weight: 900;
}


/* -----------------------------------------------------------
   RECOMMENDATIONS
----------------------------------------------------------- */

.recommendation-panel {
    margin-top: 17px;
}

.recommendation-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;

    margin-bottom: 13px;
}

.recommendation-title {
    color: #142e50;
    font-size: 21px;
    font-weight: 850;
}

.star {
    color: #f5b51b;
}

.recommendation-subtitle {
    margin-top: 3px;
    color: #71839a;
    font-size: 12px;
}

.match-badge {
    padding: 8px 13px;

    color: #31597e;
    background: #eef5fd;

    border: 1px solid #dbe7f4;
    border-radius: 20px;

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

    padding: 12px;

    border: 1px solid #dce6f1;
    border-radius: 15px;

    background: #ffffff;

    box-shadow:
        0 5px 17px rgba(32,65,101,.05);

    transition:
        transform .18s ease,
        box-shadow .18s ease;
}

.car-card:hover {
    transform: translateY(-3px);

    box-shadow:
        0 12px 25px rgba(32,65,101,.12);
}

.rank-one {
    border-color: #f1d27b;
    background: #fffdf8;
}

.rank-three {
    border-color: #e9cfc1;
    background: #fffaf7;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.rank-badge {
    width: 32px;
    height: 32px;

    display: grid;
    place-items: center;

    border-radius: 50%;

    color: #234363;
    background: #eaf2fc;

    font-size: 13px;
    font-weight: 900;
}

.rank-one .rank-badge {
    color: #704d00;
    background: #ffc83d;
}

.rank-three .rank-badge {
    color: white;
    background: #d98c55;
}

.heart {
    color: #df3b47;
    font-size: 23px;
    line-height: 1;
}

.car-image-area {
    width: 100%;
    height: 115px;

    display: flex;
    align-items: center;
    justify-content: center;

    margin-top: 3px;

    overflow: hidden;

    border-radius: 10px;
    background: #f8fbfe;
}

.car-photo {
    display: block;

    width: 100%;
    height: 115px;

    object-fit: contain;

    background: #f8fbfe;
}

.car-photo-fallback {
    width: 100%;
    height: 115px;

    display: grid;
    place-items: center;

    color: #6c87a4;
    background: #f8fbfe;

    font-size: 48px;
}

.car-name {
    min-height: 23px;

    margin-top: 7px;

    text-align: center;

    color: #183554;

    font-size: 15px;
    font-weight: 850;

    text-transform: capitalize;
}

.car-price {
    margin-top: 8px;
    padding: 8px;

    text-align: center;

    color: #1559e8;
    background: #e5f0ff;

    border-radius: 18px;

    font-size: 15px;
    font-weight: 900;
}

.car-specs {
    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 7px 5px;

    margin-top: 11px;
    padding: 0 3px 2px;

    color: #61758c;

    font-size: 10px;
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
    background: #fff5f5;
    border-color: #ffd2d2;
}

.error-text {
    margin-top: 10px;
    color: #a52d2d;
    font-weight: 700;
}


/* -----------------------------------------------------------
   FOOTER
----------------------------------------------------------- */

.footer {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 14px 38px 18px;

    color: #708299;
    font-size: 11px;
}

.footer strong {
    color: #294766;
}


/* -----------------------------------------------------------
   RESPONSIVE
----------------------------------------------------------- */

@media (max-width: 1050px) {

    .main-row {
        padding: 20px;
    }

    .cars-grid {
        grid-template-columns: 1fr;
    }

    .feature-row {
        gap: 15px;
    }
}

@media (max-width: 800px) {

    .hero {
        padding: 25px;
    }

    .hero-title {
        font-size: 31px;
    }

    .feature-row {
        flex-direction: column;
        align-items: flex-start;
    }

    .feature {
        border-right: none;
    }

    .main-row {
        padding: 15px;
    }

    .footer {
        flex-direction: column;
        gap: 8px;
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

            <h1 class="hero-title">
                Smart Car <span>Price Predictor</span>
            </h1>

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
