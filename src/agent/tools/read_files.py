from itertools import islice
from collections import deque

def read_files(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return[]
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []
    
def read_slice_of_files(file_path, lines_to_read=10, start_line=0):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
            return list(islice(lines, start_line, start_line + lines_to_read))
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return[]
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []
    
    
def read_latest_logs(file_path,max_lines):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            latest_logs = deque(file, maxlen=max_lines)
        return latest_logs
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return[]
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []