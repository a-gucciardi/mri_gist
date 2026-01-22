# taking ages and not exactly working

export SUBJECTS_DIR=/home/arnaud/projects/RESEARCH-MRI-analysis/code/volumes/recon_out
source=/home/data/dhcp_mri/sourcedata
s=001

for folder in "$source"/sub*/ses*/anat;
do
    # print helper
    # find "$folder" -maxdepth 1 -type f -not -name "*skull*" | while read -r file; do
    #     echo -e "Directory: $(dirname "$file")"
    #     echo "$(dirname "$file")" | cut -d/ -f7 | cut -d- -f2
    #     # echo "Parent Directory: $(basename $(dirname "$file"))"
    #     echo "File: $(basename "$file")"
    #     echo
    # done

    files=($(find "$folder" -maxdepth 1 -type f -not -name "*skull*" | sort))
    # track session id for subjectname
    sessid=$(echo "$(dirname "$files")" | cut -d/ -f7 | cut -d- -f2)

    # If T1 + T2 -> use recon-all with T2
    # from scandata, we know the other case only means there is no T1 -> we do not perform recon-all
    if [[ ${#files[@]} -eq 2 ]]; then
        # echo "$sessid"
        # echo "2 files"
        # echo "${files[@]}"
        # recon-all -subject $sessid -i ${files[0]} -T2 ${files[1]} -T2pial -all -parallel -openmp 8
        # quick fix : use recon-all on T1 only, on exmaples with both
        recon-all -subject $sessid -i ${files[0]} -all 
        # echo $sessid
        # rm -r $sessid/mri/transforms
    fi
done
