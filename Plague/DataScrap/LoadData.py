import re
import pandas as pd
import numpy as np
from datetime import datetime


def parse_csv_data(file_path):
    # Read the file content
    with open(file_path, 'r', encoding='latin-1') as file:
        content = file.read()
        lines = content.split('\n')

    # Initialize data structures
    sample_data = []
    applications = []
    releases_diadegma = []
    releases_trichogramma = []

    # Find the applications section
    in_applications = False
    in_releases = False
    in_sample_data = False

    for i, line in enumerate(lines):
        # Clean line
        line = line.strip()
        if not line:
            continue

        # Split by semicolons
        cells = [cell.strip() for cell in line.split(';')]

        # Track sections
        if "APLICACIONES" in line:
            in_applications = True
            in_sample_data = False
            in_releases = False
            continue

        if "LIBERACIONES" in line:
            in_applications = False
            in_sample_data = False
            in_releases = True
            continue

        if "DIA;LAPSO;FECHA_MUESTRA" in line:
            in_applications = False
            in_sample_data = True
            in_releases = False
            continue

        # Extract sample data
        if in_sample_data and cells[0].isdigit():
            try:
                dia = int(cells[0])
                fecha = cells[2]  # Store the actual date as well
                n_plantas = int(cells[3])

                # Extract dorso values
                dorso_pattern = r'\((\d+)\)_(\d+)_(\d+)_(\d+)_(\d+)'
                dorso_match = re.search(dorso_pattern, cells[4])

                if dorso_match:
                    huevos = int(dorso_match.group(1))
                    chico = int(dorso_match.group(2))
                    mediano = int(dorso_match.group(3))
                    grande = int(dorso_match.group(4))
                    pupa_no_parasitado = int(dorso_match.group(5))

                    # Calculate values
                    h_value = huevos / n_plantas
                    l_value = (chico + mediano + grande) / n_plantas
                    p_value = pupa_no_parasitado / n_plantas

                    sample_data.append({
                        'original_dia': dia,
                        'fecha': fecha,
                        'n_plantas': n_plantas,
                        'H': h_value,
                        'L': l_value,
                        'P': p_value
                    })
            except (ValueError, IndexError) as e:
                print(f"Error processing sample data on line {i + 1}: {e}")

        # Extract pesticide applications
        if in_applications and len(cells) > 3:
            # Skip the header rows
            if "FECHA APLICACIÓN" in line or not any(cells):
                continue

            # Check if this is an application row - cell[1] should be a number
            if cells[1].strip() and cells[1].strip().isdigit():
                try:
                    dia = int(cells[1])
                    fecha = cells[2]
                    insumo = cells[3]

                    # Get additional info if available
                    dosis = cells[4] if len(cells) > 4 and cells[4].strip() else ""
                    tipo = cells[5] if len(cells) > 5 and cells[5].strip() else ""

                    applications.append({
                        'original_dia': dia,
                        'fecha': fecha,
                        'insumo': insumo,
                        'dosis': dosis,
                        'tipo': tipo
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error processing application data on line {i + 1}: {e}")

        # Extract releases for beneficial insects
        if in_releases and len(cells) > 3:
            # Skip the header row
            if "FECHA LIBERACION" in line:
                continue

            # Check if this has a valid day number in cell[0]
            if cells[0].strip() and cells[0].strip().isdigit():
                try:
                    dia = int(cells[0])
                    fecha = cells[2]
                    insecto = cells[3].upper()
                    cantidad = cells[4] if len(cells) > 4 else ""

                    if 'DIADEGMA' in insecto:
                        releases_diadegma.append({
                            'original_dia': dia,
                            'fecha': fecha,
                            'cantidad': cantidad
                        })
                    elif 'TRICHOGRAMMA' in insecto:
                        releases_trichogramma.append({
                            'original_dia': dia,
                            'fecha': fecha,
                            'cantidad': cantidad
                        })
                except (ValueError, IndexError) as e:
                    print(f"Error processing release data on line {i + 1}: {e}")

    return {
        'sample_data': sample_data,
        'applications': applications,
        'releases_diadegma': releases_diadegma,
        'releases_trichogramma': releases_trichogramma
    }


def parse_date(date_str):
    """
    Parse dates in the format DD-MMM-YY (e.g., 27-jun-05)
    Returns a datetime object
    """
    month_map = {
        'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 'jun': '06',
        'jul': '07', 'ago': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dic': '12'
    }

    try:
        parts = date_str.split('-')
        if len(parts) != 3:
            print(f"Warning: Invalid date format: {date_str}")
            return None

        day = parts[0]
        month = month_map.get(parts[1].lower(), '01')  # Default to January if unknown
        year = f"20{parts[2]}" if len(parts[2]) == 2 else parts[2]

        return datetime.strptime(f"{day}-{month}-{year}", "%d-%m-%Y")
    except Exception as e:
        print(f"Error parsing date {date_str}: {e}")
        return None


def normalize_dates(data):
    """
    Normalize all dates by setting the earliest date in sample_data as day 0
    and calculating proper day numbers for all other records
    """
    # First, parse all dates
    all_dates = []

    # Parse sample data dates
    for i, record in enumerate(data['sample_data']):
        date_obj = parse_date(record['fecha'])
        if date_obj:
            data['sample_data'][i]['date_obj'] = date_obj
            all_dates.append(date_obj)
        else:
            print(f"Warning: Could not parse date for sample data: {record['fecha']}")

    # Parse application dates
    for i, record in enumerate(data['applications']):
        date_obj = parse_date(record['fecha'])
        if date_obj:
            data['applications'][i]['date_obj'] = date_obj
            all_dates.append(date_obj)
        else:
            print(f"Warning: Could not parse date for application: {record['fecha']}")

    # Parse diadegma release dates
    for i, record in enumerate(data['releases_diadegma']):
        date_obj = parse_date(record['fecha'])
        if date_obj:
            data['releases_diadegma'][i]['date_obj'] = date_obj
            all_dates.append(date_obj)
        else:
            print(f"Warning: Could not parse date for diadegma release: {record['fecha']}")

    # Parse trichogramma release dates
    for i, record in enumerate(data['releases_trichogramma']):
        date_obj = parse_date(record['fecha'])
        if date_obj:
            data['releases_trichogramma'][i]['date_obj'] = date_obj
            all_dates.append(date_obj)
        else:
            print(f"Warning: Could not parse date for trichogramma release: {record['fecha']}")

    # Find the earliest date to use as reference point (day 0)
    if all_dates:
        reference_date = min(all_dates)
        print(f"Reference date (Day 0): {reference_date.strftime('%d-%b-%y')}")

        # Calculate normalized day numbers
        for i, record in enumerate(data['sample_data']):
            if 'date_obj' in record:
                delta = (record['date_obj'] - reference_date).days
                data['sample_data'][i]['dia'] = delta

        for i, record in enumerate(data['applications']):
            if 'date_obj' in record:
                delta = (record['date_obj'] - reference_date).days
                data['applications'][i]['dia'] = delta

        for i, record in enumerate(data['releases_diadegma']):
            if 'date_obj' in record:
                delta = (record['date_obj'] - reference_date).days
                data['releases_diadegma'][i]['dia'] = delta

        for i, record in enumerate(data['releases_trichogramma']):
            if 'date_obj' in record:
                delta = (record['date_obj'] - reference_date).days
                data['releases_trichogramma'][i]['dia'] = delta
    else:
        print("Warning: No valid dates found to use as reference")

    return data


def save_to_excel(data, output_file):
    """Save the extracted data to an Excel file with multiple sheets"""
    with pd.ExcelWriter(output_file) as writer:
        # Create sample data sheet
        df_samples = pd.DataFrame(data['sample_data'])
        # Keep only necessary columns
        if 'date_obj' in df_samples.columns:
            df_samples = df_samples.drop(columns=['date_obj'])
        df_samples.to_excel(writer, sheet_name='Sample Data', index=False)

        # Create applications sheet (pesticide applications)
        df_applications = pd.DataFrame(data['applications'])
        if 'date_obj' in df_applications.columns:
            df_applications = df_applications.drop(columns=['date_obj'])
        df_applications.to_excel(writer, sheet_name='Pesticide Applications', index=False)

        # Create Diadegma releases sheet
        df_diadegma = pd.DataFrame(data['releases_diadegma'])
        if 'date_obj' in df_diadegma.columns:
            df_diadegma = df_diadegma.drop(columns=['date_obj'])
        df_diadegma.to_excel(writer, sheet_name='Diadegma Releases', index=False)

        # Create Trichogramma releases sheet
        df_trichogramma = pd.DataFrame(data['releases_trichogramma'])
        if 'date_obj' in df_trichogramma.columns:
            df_trichogramma = df_trichogramma.drop(columns=['date_obj'])
        df_trichogramma.to_excel(writer, sheet_name='Trichogramma Releases', index=False)


def get_numpy_arrays(data):
    """
    Convert the extracted data into numpy arrays for modeling

    Returns:
    - T: numpy array of time points (days) for sample data
    - E: numpy array of egg counts per plant
    - L: numpy array of larvae counts per plant
    - P: numpy array of pupae counts per plant
    - Ti: numpy array of pesticide application days
    - t_trichogramma: numpy array of days when Trichogramma was released
    - Tr_trichogramma: numpy array of Trichogramma release quantities if available
    - t_diadegma: numpy array of days when Diadegma was released
    - Tr_diadegma: numpy array of Diadegma release quantities if available
    """
    # Sort the sample data by normalized day to ensure chronological order
    sorted_data = sorted(data['sample_data'], key=lambda x: x['dia'])

    # Extract time points and measurements for sample data
    T = np.array([record['dia'] for record in sorted_data])
    E = np.array([record['H'] for record in sorted_data])  # H is eggs per plant
    L = np.array([record['L'] for record in sorted_data])  # L is larvae per plant
    P = np.array([record['P'] for record in sorted_data])  # P is pupae per plant

    # Extract pesticide application days
    sorted_applications = sorted(data['applications'], key=lambda x: x['dia'])
    Ti = np.array([app['dia'] for app in sorted_applications])

    # Extract Trichogramma release days and quantities
    sorted_trichogramma = sorted(data['releases_trichogramma'], key=lambda x: x['dia'])
    t_trichogramma = np.array([release['dia'] for release in sorted_trichogramma])

    # Try to convert quantities to numbers, use NaN if not possible
    Tr_trichogramma = np.array([
        float(release['cantidad']) if release['cantidad'] and release['cantidad'].replace('.', '', 1).isdigit()
        else np.nan
        for release in sorted_trichogramma
    ])

    # Extract Diadegma release days and quantities
    sorted_diadegma = sorted(data['releases_diadegma'], key=lambda x: x['dia'])
    t_diadegma = np.array([release['dia'] for release in sorted_diadegma])

    # Try to convert quantities to numbers, use NaN if not possible
    Tr_diadegma = np.array([
        float(release['cantidad']) if release['cantidad'] and release['cantidad'].replace('.', '', 1).isdigit()
        else np.nan
        for release in sorted_diadegma
    ])

    return T, E, L, P, Ti, t_trichogramma, Tr_trichogramma, t_diadegma, Tr_diadegma


def load_insect_data(file_path=None, from_excel=False, shift_time=None, min_day=None):
    """
    Load insect data and return numpy arrays for all required variables

    Parameters:
        file_path (str): Path to the CSV or Excel file, if None uses the default
        from_excel (bool): If True, loads from Excel, otherwise loads from CSV
        shift_time (int): If provided, subtracts this value from all time arrays
        min_day (int): If provided, filters out data before this day

    Returns:
        dict: Dictionary containing all the arrays:
            - T: time points for sample data
            - E: eggs per plant
            - L: larvae per plant
            - P: pupae per plant
            - Ti: pesticide application days
            - t_trichogramma: Trichogramma release days
            - Tr_trichogramma: Trichogramma release quantities
            - t_diadegma: Diadegma release days
            - Tr_diadegma: Diadegma release quantities
    """
    if from_excel:
        if file_path is None:
            file_path = 'extracted_insect_data.xlsx'

        # Load sample data
        df_samples = pd.read_excel(file_path, sheet_name='Sample Data')
        df_samples = df_samples.sort_values(by='dia')

        T = df_samples['dia'].to_numpy()
        E = df_samples['H'].to_numpy()
        L = df_samples['L'].to_numpy()
        P = df_samples['P'].to_numpy()

        # Load pesticide applications
        df_applications = pd.read_excel(file_path, sheet_name='Pesticide Applications')
        df_applications = df_applications.sort_values(by='dia')
        Ti = df_applications['dia'].to_numpy()

        # Load Trichogramma releases
        df_trichogramma = pd.read_excel(file_path, sheet_name='Trichogramma Releases')
        df_trichogramma = df_trichogramma.sort_values(by='dia')
        t_trichogramma = df_trichogramma['dia'].to_numpy()

        # Convert quantities to numeric, coerce errors to NaN
        Tr_trichogramma = pd.to_numeric(df_trichogramma['cantidad'], errors='coerce').to_numpy()

        # Load Diadegma releases
        df_diadegma = pd.read_excel(file_path, sheet_name='Diadegma Releases')
        df_diadegma = df_diadegma.sort_values(by='dia')
        t_diadegma = df_diadegma['dia'].to_numpy()

        # Convert quantities to numeric, coerce errors to NaN
        Tr_diadegma = pd.to_numeric(df_diadegma['cantidad'], errors='coerce').to_numpy()

    else:
        # Load from CSV
        if file_path is None:
            file_path = 'InsectData_206.csv'

        # Parse the CSV data
        data = parse_csv_data(file_path)

        # Normalize dates and recalculate day numbers
        data = normalize_dates(data)

        # Save normalized data to Excel
        excel_output = f'{file_path}_normalized.xlsx'
        save_to_excel(data, excel_output)
        print(f"Normalized data saved to {excel_output}")

        # Get numpy arrays from normalized data
        T, E, L, P, Ti, t_trichogramma, Tr_trichogramma, t_diadegma, Tr_diadegma = get_numpy_arrays(data)

    # Apply time shift if requested
    if shift_time is not None:
        T = T - shift_time
        Ti = Ti - shift_time
        t_trichogramma = t_trichogramma - shift_time
        t_diadegma = t_diadegma - shift_time

    # Filter data if min_day is provided
    if min_day is not None:
        # Filter sample data
        mask = T >= min_day
        T = T[mask]
        E = E[mask]
        L = L[mask]
        P = P[mask]

        # Filter pesticide applications
        mask_ti = Ti >= min_day
        Ti = Ti[mask_ti]

        # Filter Trichogramma releases
        mask_trich = t_trichogramma >= min_day
        t_trichogramma = t_trichogramma[mask_trich]
        Tr_trichogramma = Tr_trichogramma[mask_trich]

        # Filter Diadegma releases
        mask_diad = t_diadegma >= min_day
        t_diadegma = t_diadegma[mask_diad]
        Tr_diadegma = Tr_diadegma[mask_diad]

    # Return all arrays in a dictionary for easy access
    return {
        'T': T,
        'E': E,
        'L': L,
        'P': P,
        'Ti': Ti,
        't_trichogramma': t_trichogramma,
        'Tr_trichogramma': Tr_trichogramma,
        't_diadegma': t_diadegma,
        'Tr_diadegma': Tr_diadegma
    }


def main():
    input_file = 'InsectData_206.csv'
    output_file = 'normalized_insect_data.xlsx'

    # Parse the CSV data
    data = parse_csv_data(input_file)

    # Normalize dates and recalculate day numbers
    normalized_data = normalize_dates(data)

    # Save normalized data to Excel
    save_to_excel(normalized_data, output_file)
    print(f"Normalized data saved to {output_file}")

    # Get numpy arrays from normalized data
    arrays = load_insect_data(input_file)

    # Print the arrays
    print("\nNumPy Arrays:")
    print(f"T = {repr(arrays['T'])}")
    print(f"E = {repr(arrays['E'])}")
    print(f"L = {repr(arrays['L'])}")
    print(f"P = {repr(arrays['P'])}")
    print(f"Ti (Pesticide applications) = {repr(arrays['Ti'])}")
    print(f"t_trichogramma (Release days) = {repr(arrays['t_trichogramma'])}")
    print(f"Tr_trichogramma (Release quantities) = {repr(arrays['Tr_trichogramma'])}")
    print(f"t_diadegma (Release days) = {repr(arrays['t_diadegma'])}")
    print(f"Tr_diadegma (Release quantities) = {repr(arrays['Tr_diadegma'])}")


if __name__ == "__main__":
    main()