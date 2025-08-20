# celebrity-look-alike

Experimenting image recognition tools and packages.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `example.env` to `.env` and adjust any paths or ports as needed.
3. (Optional) Preload celebrity embeddings using the helpers:
   ```bash
   python - <<'PYTHON'
from libs.utils.ml_model.src.helpers import image_predict_helpers
image_predict_helpers.generate_and_store_embeddings_from_celeb_names(["Tom Cruise"])
PYTHON
   ```

## Run

Launch the FastAPI backend and Streamlit interface:

```bash
./run_app.sh
```

The API runs on `http://localhost:4549` and the web UI on `http://localhost:4550`.

## Usage

Open the web page and either provide an image URL, upload a picture, or take one with your webcam to find your celebrity twin.
