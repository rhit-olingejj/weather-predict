import pytest
import pandas as pd
from datetime import datetime
from export_data import get_cutoff_date_str, build_query, save_dataframe


def test_get_cutoff_date_str():
    """
    test that the date string is formatted correctly for SQL
    """
    date_str = get_cutoff_date_str(5)
    
    # should be formatted as YYYY-MM-DD HH:MM:SS (19 characters)
    assert len(date_str) == 19
    assert "-" in date_str
    assert ":" in date_str
    
    # check successfully parses back to a datetime object
    try:
        datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        pytest.fail(f"Date string {date_str} is not in the expected SQL format.")


def test_build_query():
    """
    test that the SQL query is constructed with the requested timeframe
    """
    query = build_query(10)
    
    assert "SELECT" in query
    assert "FROM WX_DATA w" in query
    assert "WHERE w.dtg >=" in query
    assert "ORDER BY w.dtg DESC" in query


def test_save_dataframe_csv(tmp_path):
    """
    test exporting a dummy dataframe to CSV using pytest's temporary directory
    """
    # mock dataframe
    df = pd.DataFrame({
        "city_name": ["New York", "London"],
        "temp_C": [19.1, 15.0]
    })
    
    output_file = tmp_path / "test_export.csv"
    
    save_dataframe(df, "csv", str(output_file))
    
    # verify file was created and contains expected data
    assert output_file.exists()
    content = output_file.read_text()
    assert "city_name,temp_C" in content
    assert "New York,19.1" in content


def test_save_dataframe_json(tmp_path):
    """
    test exporting a dummy dataframe to JSON
    """
    df = pd.DataFrame({
        "city_name": ["Portland"],
        "temp_C": [22.5]
    })
    
    output_file = tmp_path / "test_export.json"
    
    save_dataframe(df, "json", str(output_file))
    
    # verify file was created and contains expected data
    assert output_file.exists()
    content = output_file.read_text()
    assert "Portland" in content
    assert "22.5" in content
