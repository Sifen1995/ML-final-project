# ML Project



## 👥 Group Members
1. **Sifen Getachew** — UGR/2452/15
2. **Girma Enkuchile** — UGR/8130/15
3. **Yitbarek Alemu** — UGR/7554/15
4. **Blen Nigussie** — UGR/3480/15
5. **Abiy Aragie** — UGR/8104/15


A small machine learning project using the `student-mat.csv` dataset.

## Description
This project implements an end-to-end Machine Learning pipeline to predict students' final academic grades (G3). Using the UCI Student Performance dataset, we analyze demographic, social, and academic features to identify key drivers of success and build predictive models to assist educational interventions.


## Setup
1. Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
# or install scikit-learn directly:
python3 -m pip install scikit-learn pandas numpy
```

## Run
Start the app:

```bash
python app.py
```

## Files
- `student-mat.csv` — dataset
- `app.py` — main script
- `requirements.txt` — (optional) dependency list

## Notes
Replace `venv` with your preferred environment name. Use `conda` if you prefer conda environments:

```bash
conda create -n myenv python=3.10
conda activate myenv
```