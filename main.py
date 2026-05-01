import os
import shutil
import pandas as pd
import flirimageextractor
from datetime import datetime

def process_thermal_directory(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)
    extractor = flirimageextractor.FlirImageExtractor()
    
    excel_path = None
    for file in os.listdir(source_folder):
        if file.endswith('.xls') or file.endswith('.xlsx'):
            excel_path = os.path.join(source_folder, file)
            break
            
    climate_data = None
    if excel_path:
        climate_data = pd.read_excel(excel_path, sheet_name='List')
        climate_data['Time'] = pd.to_datetime(climate_data['Time'])

    for file in os.listdir(source_folder):
        if file.endswith('.jpg'):
            full_path = os.path.join(source_folder, file)

            parts = file.split('__')
            if len(parts) < 2:
                continue
                
            capture_number = parts[0]
            date_text = parts[1].replace('.jpg', '')
            
            folder_name = f"snapshot_{capture_number}"
            capture_route = os.path.join(destination_folder, folder_name)
            os.makedirs(capture_route, exist_ok=True)

            extractor.process_image(full_path)
            thermal_matrix = extractor.get_thermal_np()

            df_matrix = pd.DataFrame(thermal_matrix)
            df_matrix.to_csv(os.path.join(capture_route, 'thermal_matrix.csv'), index=False, header=False)
            
            shutil.copy(full_path, os.path.join(capture_route, file))

            if climate_data is not None:
                capture_time = datetime.strptime(date_text, '%d-%m-%Y_%H-%M-%S-%f')
                time_differences = abs(climate_data['Time'] - capture_time)
                optimal_index = time_differences.idxmin()
                
                df_environmental = pd.DataFrame([climate_data.loc[optimal_index]])
                
                if 'No.' in df_environmental.columns:
                    df_environmental = df_environmental.drop(columns=['No.'])
                    
                if 'Time' in df_environmental.columns:
                    df_environmental['Time'] = df_environmental['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                df_environmental.to_csv(os.path.join(capture_route, 'environmental_data.csv'), index=False)

def process_all_experiments(base_raw_folder, base_processed_folder):
    if not os.path.exists(base_raw_folder):
        return

    for folder_name in os.listdir(base_raw_folder):
        source_folder = os.path.join(base_raw_folder, folder_name)
        
        if os.path.isdir(source_folder):
            destination_folder = os.path.join(base_processed_folder, folder_name)
            
            if os.path.exists(destination_folder):
                continue
                
            process_thermal_directory(source_folder, destination_folder)

if __name__ == '__main__':
    base_raw_dir = 'raw_data'
    base_processed_dir = 'processed_data'
    
    process_all_experiments(base_raw_dir, base_processed_dir)