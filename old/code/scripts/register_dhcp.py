import ants
from glob import glob
from pathlib import Path
from natsort import natsorted, natsort_keygen

import subprocess, os, time
import tqdm
import pandas as pd


sourcepath = '/Users/arnaud/Downloads/sourcedata_orig'
path_templates = '/Users/arnaud/Documents/GitHub/dhcp-volumetric-atlas-groupwise/mean'

# dhcp_T1ws = natsorted(glob(f"{sourcepath}/sub*/ses*/anat/*T1w.*"))
# dhcp_T2ws = natsorted(glob(f"{sourcepath}/sub*/ses*/anat/*T2w.*"))
# dhcp_all = natsorted(glob(f"{sourcepath}/sub*/ses*/anat/*"))


def main(test = False):
    input_data = natsorted(glob(f"{sourcepath}/sub*/ses*/anat/*"))
    print(f"Total dhcp :{len(input_data)}")
    # logging
    outputdir = "../dhcp_register/"
    log = f"{outputdir}/register_log.txt"
    Path(log).touch(exist_ok=True)
    os.makedirs(outputdir, exist_ok=True)

    # csv aligning, given a scan we want to register but with the right type of MRI
    csvpath = '/Users/arnaud/Downloads/sourcedata_orig/participants.tsv'
    ds_info = pd.read_csv(csvpath, sep ='\t')
    sorted_ds = ds_info.sort_values(by="participant_id", key=natsort_keygen())

    with open(log, "r") as f:
        processed_files = set(f.read().splitlines())

    print(f"Already started files : {len(processed_files)}/{len(input_data)}.")
    print(f"{processed_files}")
    before = natsorted(glob(f"{outputdir}/*.nii.gz"))
    print(f"Already Completed files {len(processed_files)}/{len(input_data)}. \n")

    if not test:
        with open(log, "a") as logfile:
            for i, file in enumerate(tqdm.tqdm(input_data[:4])):
                # folder structure check
                parts = file.split("/")
                subject_id = parts[-4] # sub-XXXX
                session_id = parts[-3] # ses-XX
                img = parts[-1]

                age = sorted_ds[sorted_ds['participant_id'] == subject_id[4:]]['birth_age'].astype('int').values[0]
                template_path = path_templates + f'/ga_{age}' + f'/template_t{img[-9]}.nii.gz'

                # print(age, template_path)
                output_filename = f"{img[:-7]}_registered.nii.gz"
                output_path = f"{outputdir}/{output_filename}"
                # print(output_filename)

                # check if already processed
                if f"{img[:-7]}" in processed_files and os.path.exists(output_path):
                    print(f"Skipping already processed and logged : {subject_id} {session_id} \n")
                    continue

                # ants registration
                try:
                    print(f"Processing and logging {subject_id} {session_id} located in : {file}")
                    input_image = ants.image_read(file)
                    template_image = ants.image_read(template_path)
                    print(f"Scan age detected {age} => using template file {template_path}")
                    # Perform registration with SyN algorithm
                    transformed_image = ants.registration(fixed=template_image, moving=input_image, type_of_transform='SyN')
                    ants.image_write(transformed_image['warpedmovout'], f"{output_path}")

                    logfile.write(f"{img[:-7]}\n")
                    print(f"Saved registered image to {os.path.abspath(output_path)}. \n")

                except KeyboardInterrupt:
                    print("\nProcessing interrupted. Progress has been saved.")
                    break

                except Exception as e:
                    print(f"Error processing {subject_id} {session_id}: {str(e)}")

    after = natsorted(glob(f"{outputdir}/*.nii.gz"))
    print(f"Newly completed files {len(after) - len(before)}")

if __name__ == '__main__':
    main(test=False)
