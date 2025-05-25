
# 🌳 street_tree

Build a deep learning model to classify tree health (`Good`, `Fair`, `Poor`) using data from the NYC 2015 Street Tree Census.

---

## 📌 Project Description

This project involves the development and training of a deep learning model to classify the health condition of street trees based on data from the 2015 New York City Street Tree Census.

The model predicts one of the following conditions:
- **Good**
- **Fair**
- **Poor**

---

## 🧠 Tree Health Prediction API

### Endpoint: `/predict_health/`

The API allows you to send tree data in JSON format and receive a prediction of tree health.

- **Method:** `POST`  
- **URL:** `http://127.0.0.1:8000/predict_health/`  
- **Description:** Returns the predicted health condition based on input JSON.

### 🔻 Sample Request Body:
```json
[
  {
    "tree_id": 536325,
    "block_id": 415591,
    "tree_dbh": 56,
    "curb_loc": 0,
    "spc_latin": "Fraxinus",
    "steward": "None",
    "guards": "Unsure",
    "sidewalk": 0,
    "user_type": "NYC Parks Staff",
    "problems": "NoProblem",
    "root_stone": 0,
    "root_grate": 0,
    "root_other": 0,
    "trunk_wire": 0,
    "trnk_light": 0,
    "trnk_other": 0,
    "brch_light": 0,
    "brch_shoe": 0,
    "brch_other": 0,
    "postcode": 10306,
    "borough": "Staten Island",
    "cncldist": 50,
    "st_assem": 62,
    "st_senate": 24,
    "nta": "SI24",
    "boro_ct": 5027900,
    "latitude": 40.571808,
    "longitude": -74.14425163
  }
]
```

### 🔺 Sample Response:
```json
[
  {
    "tree_id": 536325,
    "predictions": "Good",
    "probably": [0.1, 0.3, 0.6],
    "class_labels": "Good",
    "name_model": "TreeHealthModel"
  }
]
```

### Response Fields:
- `tree_id`: Unique tree identifier.
- `predictions`: Predicted tree condition (`Good`, `Fair`, `Poor`).
- `probably`: Class probabilities in the order `[Poor, Fair, Good]`.
- `class_labels`: Label of the predicted class.
- `name_model`: Name of the model used for prediction.

### 📈 Interpretation Example:
For the tree with `tree_id = 536325`, the model predicted:
- **Poor**: 10%  
- **Fair**: 30%  
- **Good**: 60%

---

## 📁 Project Structure

```
street_tree/
├── data/                     # Project data
│   ├── raw/                  # Raw data
│   ├── processed/            # Processed data
├── models/                   # Trained model files
├── notebook/                 # Jupyter notebooks
│   ├── eda.ipynb             # Exploratory Data Analysis (EDA)
│   ├── test_api.ipynb        # API testing
├── src/                      # Project source code
│   ├── config.py             # Configuration
│   ├── downloader.py         # Data/model loading scripts
│   ├── modeling.py           # Model architecture definition
│   ├── preprocessing.py      # Data preprocessing scripts
│   ├── utils.py              # Utility functions
│   ├── model.py              # Pydantic models for FastAPI
├── main.py                   # Main file to run the FastAPI app
├── README.md                 # Project documentation
```

### 🔍 Key Files:
- `notebook/eda.ipynb`: Exploratory Data Analysis notebook.
- `notebook/test_api.ipynb`: Notebook for testing the API.
- `main.py`: Launches the FastAPI server and defines the `/predict_health/` endpoint.
- `src/modeling.py`: Defines the deep learning model architecture.
- `src/preprocessing.py`: Prepares and processes input data.

---

## 🚀 How to Run the Project

1. **Install dependencies:**
   ```bash
   pip install -r requirements_venv.txt
   ```

2. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload
   ```

3. **Open the API docs in your browser:**
   [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ✅ Conclusion

This project provides an API for predicting the health condition of trees using structured census data.  
The model demonstrates promising performance and can be further improved by refining the data, enhancing the architecture, and exploring advanced machine learning techniques.

**Model AUC-ROC on out-of-time (OOT) sample:** `0.703`
