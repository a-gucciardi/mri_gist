# sample to loop dhcp files

source=/home/data/dhcp_mri/sourcedata_orig
# source=/Users/arnaud/Downloads/sourcedata_orig
# source=/Users/arnaud/Downloads/sourcedata_not2

total_folders=0
total_files=0
t1_count=0
t2_count=0

for folder in "$source"/sub*/ses*/anat;
do
    files=($(find "$folder" -maxdepth 1 -type f -not -name "*skull*" | sort))
    echo "$files"
    sessid=$(echo "$(dirname "$files")" | cut -d/ -f7 | cut -d- -f2)
    # If T1 + T2 
    if [[ ${#files[@]} -eq 2 ]]; then
        total_files=$((total_files + ${#files[@]}))
    else
        # count T1 only vs T2 only
        if [[ $(basename "$files") == *T1* ]]; then 
            t1_count=$((t1_count + 1))
        elif [[ $(basename "$files") == *T2* ]]; then 
            t2_count=$((t2_count + 1))
            echo "Folder with T2 only: $folder"
        fi
    fi
    total_folders=$((total_folders+1))
done

# Counts
echo "Total folders: $total_folders"
echo "Total T1 + T2: $total_files"
echo "Unique T1 files: $t1_count"
echo "Unique T2 files: $t2_count"
