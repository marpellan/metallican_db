import pandas as pd
import matplotlib.pyplot as plt

### NRCan dataset
def plot_commodity_distribution(df, primary_col, group_col, title_prefix):
    """
    Function to plot commodity distribution for a given dataset.

    Parameters:
    - file_path: str, path to the Excel file
    - sheet_name: str, sheet name to read from the Excel file
    - primary_col: str, column containing the list of commodities (first one will be taken)
    - group_col: str, column containing the commodity group classification
    - title_prefix: str, prefix to differentiate plots (e.g., "Producing Mines" or "Metal Works")
    """

    # Extract first commodity
    df["Primary_Commodity"] = df[primary_col].fillna("N/A").str.split(",").str[0]
    primary_commodity_counts = df["Primary_Commodity"].value_counts()

    # Extract commodity group, including N/A
    df["Commodity_Group"] = df[group_col].fillna("N/A")
    commodity_group_counts = df["Commodity_Group"].value_counts()

    # Plot bar chart for primary commodities
    plt.figure(figsize=(12, 6))
    plt.bar(primary_commodity_counts.index, primary_commodity_counts.values)
    plt.xticks(rotation=90)
    plt.xlabel("Primary Commodity")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} - Number by Primary Commodity")
    plt.show()

    # Plot pie chart for commodity groups
    plt.figure(figsize=(6, 6))
    plt.pie(commodity_group_counts, labels=commodity_group_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f"{title_prefix} - Distribution by Commodity Group")
    plt.show()

### Dallaire-Fortin (2024) dataset
def is_mine_active(row):
    '''
    For the dataset of Dallaire-Fortin

    Function to define if the mine is currently active,
    based on the Changes in Status of Production
    '''

    current_year = 2022

    # Convert year values to integers, ignoring non-numeric values
    def to_int(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    # Convert all year columns to integers
    open1 = to_int(row['open1'])
    close1 = to_int(row['close1'])
    open2 = to_int(row['open2'])
    close2 = to_int(row['close2'])
    open3 = to_int(row['open3'])
    close3 = to_int(row['close3'])

    # Check if any of the 'close' columns have the value 'open'
    if row['close1'] == 'open' or row['close2'] == 'open' or row['close3'] == 'open':
        return 'Active'

    # Find the latest year among open and close columns
    years = [open1, close1, open2, close2, open3, close3]
    years = [year for year in years if year is not None]

    if not years:
        return 'Unknown'

    latest_year = max(years)

    # If the latest year is a 'close' year, the mine is inactive
    if latest_year in [close1, close2, close3]:
        return 'Inactive'

    # If the latest year is an 'open' year and it's the current year or later, consider it active
    if latest_year in [open1, open2, open3] and latest_year >= current_year:
        return 'Active'

    # For all other cases, consider it inactive
    return 'Inactive'


# Function to combine open-close periods into a single column and drop the open/close columns
def combine_open_close_periods(df):
    def get_periods(row):
        periods = []
        for i in range(1, 4):  # Since we have open1-close1, open2-close2, open3-close3
            open_col = f"open{i}"
            close_col = f"close{i}"
            open_year = row[open_col]
            close_year = row[close_col]
            if pd.notnull(open_year):
                if close_year == "open":
                    periods.append(f"{int(open_year)}-Present")
                elif pd.notnull(close_year):
                    periods.append(f"{int(open_year)}-{int(close_year)}")
        return ", ".join(periods)

    # Apply the function to each row to create the Operating_Periods column
    df["operating_periods"] = df.apply(get_periods, axis=1)

    # Drop the open/close columns
    open_close_cols = [f"open{i}" for i in range(1, 4)] + [f"close{i}" for i in range(1, 4)]
    df = df.drop(columns=open_close_cols, errors='ignore')

    return df


### GHG dataset
def filter_ghg_facility_naics(df, classifications):
    # Normalize the classifications to lowercase for case-insensitive comparison
    classifications_lower = [cls.lower() for cls in classifications]

    df_copy = df.copy()
    df_copy['NAICS_Lower'] = df_copy['Industry classification'].str.lower()
    filtered_df = df_copy[df_copy['NAICS_Lower'].isin(classifications_lower)]
    filtered_df = filtered_df.drop(columns=['NAICS_Lower'])
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df


### NPRI dataset
def clean_npri(excel_path: str,
               sheet_name_data: str = "INRP-NPRI 2023",
               sheet_name_mapping: str = "mapping_emissions") -> pd.DataFrame:
    """
    Extracts SQL-friendly long-format NPRI data using a structured mapping.

    - Loads data with skiprows=3 to get clean metadata.
    - Renames emission columns using Emission_type + Sub_emission_type_EN.
    - Melts emission columns into long format.
    - Keeps all metadata columns untouched.
    - Removes rows where value is NaN or zero.

    Args:
        excel_path (str): Path to the Excel file.
        sheet_name_data (str): Sheet with the NPRI data.
        sheet_name_mapping (str): Sheet with emission column metadata.

    Returns:
        pd.DataFrame: SQL-ready long-format DataFrame.
    """
    import pandas as pd

    # Step 1: Load the emission column mapping
    mapping_df = pd.read_excel(excel_path, sheet_name=sheet_name_mapping)
    mapping_df["Emission_type"] = mapping_df["Emission_type"].fillna(method="ffill")
    mapping_df["unique_column_name"] = (
        mapping_df["Emission_type"].str.strip() + " - " + mapping_df["Sub_emission_type_EN"].str.strip()
    )

    # Step 2: Load the NPRI data with clean headers
    df = pd.read_excel(excel_path, sheet_name=sheet_name_data, skiprows=3)

    # Step 3: Identify emission columns (based on the mapping length)
    n_emission_cols = len(mapping_df)
    emission_cols = df.columns[-n_emission_cols:]

    # Step 4: Rename only emission columns
    rename_map = dict(zip(emission_cols, mapping_df["unique_column_name"]))
    df_renamed = df.rename(columns=rename_map)

    # Step 5: Melt the emission columns
    id_vars = [col for col in df.columns if col not in emission_cols]
    melted = df_renamed.melt(id_vars=id_vars,
                             value_vars=rename_map.values(),
                             var_name="full_emission_column",
                             value_name="value")

    # Step 6: Split into emission_type and emission_subtype
    melted[["emission_type", "emission_subtype"]] = melted["full_emission_column"].str.split(" - ", n=1, expand=True)

    # Step 7: Remove NaNs and zeros
    melted = melted.dropna(subset=["value"])
    melted = melted[melted["value"] != 0]

    return melted.drop(columns=["full_emission_column"])


### SUT dataset
def clean_sut(df, year=None, naics=None, units_to_exclude=None):
    # Drop specified columns
    columns_to_remove = ['DGUID', 'UOM_ID', 'SCALAR_ID', 'VECTOR', 'COORDINATE', 'STATUS', 'SYMBOL', 'TERMINATED',
                         'DECIMALS', 'SCALAR_FACTOR']
    df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')

    # Filter by year if specified
    if year is not None:
        df_cleaned = df_cleaned[df_cleaned['REF_DATE'] == year]

    # Filter by NAICS if specified
    if naics is not None:
        df_cleaned = df_cleaned[df_cleaned['North American Industry Classification System (NAICS)'] == naics]

    # Exclude specified units from the UOM column if provided
    if units_to_exclude is not None:
        df_cleaned = df_cleaned[~df_cleaned['UOM'].isin(units_to_exclude)]

    # Remove rows where VALUE is 0 or NaN
    df_cleaned = df_cleaned[df_cleaned['VALUE'].notna() & (df_cleaned['VALUE'] != 0)]

    df_cleaned.reset_index(drop=True, inplace=True)

    return df_cleaned


def split_by_naics(df_cleaned):
    # Group the DataFrame by the 'NAICS' column
    naics_groups = df_cleaned.groupby('North American Industry Classification System (NAICS)')

    # Create a dictionary where keys are NAICS values and values are the corresponding DataFrames
    naics_dfs = {naics: group.reset_index(drop=True) for naics, group in naics_groups}

    return naics_dfs