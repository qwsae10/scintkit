from pathlib import Path
import shutil
import subprocess as sp
import time 

def binary_to_clean_txt(binary_file, work_dir=None, output_dir=None, keep_raw_txt=False, overwrite=False, verbose=True):
    #Method to convert Scintpi binary files to cleaned txt files.
    '''
    Converts a Scintpi binary file to a cleaned text file.

    Args:
        binary_file (str): The path to the binary file.
        work_dir (str, optional): The directory to use for temporary files. Defaults to the current working directory.
        output_dir (str, optional): The directory to save the output file. Defaults to the current working directory.
        keep_raw_txt (bool, optional): Whether to keep the raw text file. Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Path: The path to the cleaned text file.
    '''

    binary_file = Path(binary_file).resolve()

    if not binary_file.exists():
        raise FileNotFoundError(binary_file)
    

    if output_dir is None:
        output_dir = Path.cwd()

    if work_dir is None:
        work_dir = output_dir

    output_dir = Path(output_dir).resolve()
    work_dir = Path(work_dir).resolve()

    working_binary = work_dir / binary_file.name    
 

    if not working_binary.exists():
        shutil.copy2(binary_file, working_binary)    

    commands = ["rxtools_exec.sh", "bin2asc", "-f", working_binary.name, "--extractGenMeas"]
    result = sp.run(commands, capture_output=True, text=True, cwd=work_dir)
    
    if result.returncode != 0:
        raise RuntimeError(f"Error occurred while converting {working_binary} to text: {result.stderr}")
    

    txt_files = list(
    work_dir.glob(
        f"{working_binary.name}*measurements.txt"
    ))

    if not txt_files:
        raise FileNotFoundError(
        "No measurements file produced by RXTools"
    )

    measurement_file = txt_files[0]

    txt_file = output_dir / (
    measurement_file.stem + ".txt"
    )

    return txt_file


