import os
import subprocess
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
from tqdm import tqdm

def process_file(input_file, input_dir, output_dir, temp_dir, ftetwild_path, mattopo_path):
    """
    Processes a single file through the two-step conversion pipeline.
    1. input_file -> temp.msh (using fTetWild)
    2. temp.msh -> output.ply (using MATTOPO)
    """
    try:
        # Preserve the relative path from the input_dir
        relative_path = os.path.relpath(os.path.dirname(input_file), input_dir)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Create a unique name for the temp file to avoid collisions
        unique_temp_name = f"{relative_path.replace(os.sep, '_')}_{base_name}"
        temp_msh_file = os.path.join(temp_dir, f"{unique_temp_name}.msh")

        # Define the final output directory, preserving the structure
        final_output_dir = os.path.join(output_dir, relative_path)
        os.makedirs(final_output_dir, exist_ok=True)

        # Step 1: Convert to .msh using FloatTetwild_bin
        ftetwild_cmd = [
            ftetwild_path,
            "-i", input_file,
            "-o", temp_msh_file
        ]
        # print(f"Running: {' '.join(ftetwild_cmd)}") # Quieter output
        result1 = subprocess.run(ftetwild_cmd, capture_output=True, text=True, check=True)

        # Step 2: Convert .msh to .ply using MATTOPO
        mattopo_cmd = [
            mattopo_path,
            temp_msh_file,
            "3000",
            "cad",
            "0.1",
            final_output_dir # Use the structured output directory
        ]
        # print(f"Running: {' '.join(mattopo_cmd)}") # Quieter output
        result2 = subprocess.run(mattopo_cmd, capture_output=True, text=True, check=True)
        
        output_ply_path = os.path.join(final_output_dir, f"{base_name}.ply")
        return f"Successfully processed {input_file} -> {output_ply_path}"
    except subprocess.CalledProcessError as e:
        error_message = f"Failed to process {input_file}.\n"
        error_message += f"Command: {' '.join(e.cmd)}\n"
        error_message += f"Return Code: {e.returncode}\n"
        error_message += f"Stdout: {e.stdout}\n"
        error_message += f"Stderr: {e.stderr}\n"
        return error_message
    except Exception as e:
        return f"An unexpected error occurred while processing {input_file}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Batch process 3D files with fTetWild and MATTOPO.")
    parser.add_argument("input_dir", help="Directory containing input files (*.off, *.stl, *.obj)")
    parser.add_argument("output_dir", help="Directory to save the output *.ply files")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(), help="Maximum number of parallel processes. Defaults to the number of CPU cores.")
    
    args = parser.parse_args()

    # --- Configuration: Set paths to your executables ---
    # Assuming the script is run from the MATTopo project root
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ftetwild_path = os.path.join(script_dir, "fTetWild/build/FloatTetwild_bin")
    mattopo_path = os.path.join(script_dir, "build/bin/MATTOPO")

    # Check if executables exist
    if not os.path.exists(ftetwild_path):
        print(f"Error: fTetWild executable not found at {ftetwild_path}")
        return
    if not os.path.exists(mattopo_path):
        print(f"Error: MATTOPO executable not found at {mattopo_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all relevant files recursively
    print(f"Recursively searching for files in {args.input_dir}...")
    file_types = ["**/*.off", "**/*.stl", "**/*.obj"]
    files_to_process = []
    for file_type in file_types:
        # Use recursive=True with glob to find files in subdirectories
        search_path = os.path.join(args.input_dir, file_type)
        files_to_process.extend(glob.glob(search_path, recursive=True))

    if not files_to_process:
        print(f"No matching files found in {args.input_dir} and its subdirectories.")
        return

    print(f"Found {len(files_to_process)} files to process.")
    print(f"Using a maximum of {args.max_workers} parallel processes.")

    with tempfile.TemporaryDirectory() as temp_dir:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(process_file, f, args.input_dir, args.output_dir, temp_dir, ftetwild_path, mattopo_path) for f in files_to_process]
            
            for future in tqdm(as_completed(futures), total=len(files_to_process), desc="Processing files"):
                result = future.result()
                if "Failed" in result or "Error" in result:
                    tqdm.write(result) # Print errors to the console without disturbing the progress bar

    print("Batch processing complete.")
    print(f"Output files are saved in {args.output_dir}")

if __name__ == "__main__":
    main()