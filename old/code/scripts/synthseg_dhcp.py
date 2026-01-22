from glob import glob
from pathlib import Path
from natsort import natsorted
import subprocess, os

# from synthseg_loop_v2.py
# To use after registration
# Different folder structure at the moment
""" V2 for registered outputs, to change when running on server vs local"""

test = False
device = "cpu"

dhcp_reg_datadir = "../dhcp_register" 
input_data = natsorted(glob(f"{dhcp_reg_datadir}/*.nii.gz"))
outputdir = "../dhcp_segmentations"
print(f"Total dhcp registered :{len(input_data)}")

csvdir = f"{outputdir}/csv"
logdir = f"{csvdir}/synthseged_files.txt"
os.makedirs(outputdir, exist_ok=True)
os.makedirs(csvdir, exist_ok=True)
Path(logdir).touch(exist_ok=True)

with open(logdir, "r") as f:
    processed_files = set(f.read().splitlines())

print(f"Already started files : {len(processed_files)}/{len(input_data)}.")
if test : print(f"{processed_files}")
before = natsorted(glob(f"{outputdir}/*.nii.gz"))
print(f"Already completed files {len(processed_files)}/{len(input_data)}. \n")

if not test:
    with open(logdir, "a") as logfile:
        for file in input_data[:]:
            # folder structure check
            scan_filename = file.split("/")[-1]
            subject_id = scan_filename.split("_")[0] # sub-XXXX
            session_id = scan_filename.split("_")[1] # ses-XX
            img = scan_filename.split("_")[2]        # *.nii.gz
            # print(scan_filename, subject_id, session_id, img)

            # if after reg:
            name = scan_filename.split("_registered")[0]
            output_filename = f"{name}_synthseg_robust_{device}.nii.gz"
            # print(name, output_filename)
            # elif source: (unused)
            # output_filename = f"{subject_id}_{session_id}_synthseg_robust_cpu_mac.nii.gz"

            output_path = f"{outputdir}/{output_filename}"
            csv_path = f"{csvdir}/{name}_vol.csv"
            qc_path = f"{csvdir}/{name}_qc.csv"
            # print(output_path, csv_path)

            # check if already processed and csvs present
            if f"{output_filename}" in processed_files and os.path.exists(output_path):
                pass
                if os.path.exists(csv_path) and os.path.exists(qc_path):
                    print(f"Skipping already processed and logged : {output_filename} \n")
                    continue
                else:
                    print(f"Processed but not logged : {output_filename} \n")

            # synthseg
            try:
                print(f"Processing and logging {output_filename} : ")
                print(f"{file}")
                # parcellation
                # cmd_parc = f'mri_synthseg --i {file} --o {output_path} --vol {csv_path} --qc {qc_path} --parc --robust --cpu --threads 8'
                # no parcellation
                cmd_no_parc = f'mri_synthseg --i {file} --o {output_filename} --vol {csv_path} --qc {qc_path} --robust --cpu --threads 8'
                # !! Assumes freesurfer is on
                seg_process = subprocess.run([cmd_no_parc], shell = True)

                logfile.write(f"{output_filename}\n")
                print(f"Saved synthseg to {os.path.abspath(output_path)}. \n")

            except KeyboardInterrupt:
                print("\nProcessing interrupted. Progress has been saved.")
                break

            except subprocess.CalledProcessError as e:
                print(f"Error processing {output_filename}: {str(e)}")

    after = natsorted(glob(f"{outputdir}/*.nii.gz"))
    print(f"Newly completed files {len(after) - len(before)}")

if test: 
    print("Files to segment : ")
    print(input_data)