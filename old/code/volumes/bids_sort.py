import os
import shutil

def copy_bids_folder(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for subject in os.listdir(src_folder):
        subject_path = os.path.join(src_folder, subject)
        if os.path.isdir(subject_path):
            t2w_only = True
            for root, _, files in os.walk(subject_path):
                for file in files:
                    if not file.endswith("T2w.nii.gz"):
                        t2w_only = False
                        break
                if not t2w_only:
                    break

            if not t2w_only:
                dest_subject_path = os.path.join(dest_folder, subject)
                shutil.copytree(subject_path, dest_subject_path)

source_folder = "/Users/arnaud/Downloads/sourcedata"
destination_folder = "/Users/arnaud/Downloads/sourcedata_not2"
copy_bids_folder(source_folder, destination_folder)
