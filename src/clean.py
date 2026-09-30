import pandas as pd
import re
from pathlib import Path

def clean_data():
    pass

def save_data(data = pd.DataFrame | None)->None:
    """Saves the cleaned data."""
    pass

def load_data()->pd.DataFrame:
    data_path = Path(__file__).resolve().parent.parent / "data" / "wta.csv"
    return pd.read_csv(data_path)

def main():
    records = clean_data(load_data())
    save_data(records)
    

if __name__ == "__main__":
    main()
