#!/bin/bash

DATA_DIR="data/genshin-impact"
EXTRACT_PATH="$DATA_DIR/extracted"
mkdir -p $DATA_DIR
mkdir -p $EXTRACT_PATH

download_and_unzip() {
    SAVE_PATH="$DATA_DIR/$2"

    if [ ! -f "$SAVE_PATH" ]; then
        curl -L $1 --output $SAVE_PATH
    fi
    unzip -n $SAVE_PATH -d $EXTRACT_PATH
}

remove_nsfw_images() {
    nsfw_tags=("nude" "completely nude" "topless" "bottomless" "sex" "oral" "fellatio gesture" "tentacle sex",
             "nipples" "pussy" "vaginal" "pubic hair" "anus" "ass focus" "penis" "cum" "condom" "sex toy")
    prompt_files=$(find "$EXTRACT_PATH" -type f -name "*.txt")

    for file in $prompt_files; do
        IFS=', ' read -ra tags <<< "$(cat "$file")"
        remv=false

        for tag in "${tags[@]}"; do
            for nsfw_tag in "${nsfw_tags[@]}"; do
                if [[ "$tag" == "$nsfw_tag" ]]; then
                    remv=true
                    break
                fi
            done
            if [[ $remv == true ]]; then
                break
            fi
        done

        if [[ $remv == true ]]; then
            img_path="${file%.*}.jpg"
            echo "${img_path} contains nsfw content! Removing..."

            rm $img_path $file
        fi
    done
}

download_and_unzip https://huggingface.co/datasets/animelover/genshin-impact-images/resolve/main/data/data-0000.zip data-000.zip $DATA_DIR
# download_and_unzip https://huggingface.co/datasets/animelover/genshin-impact-images/resolve/main/data/data-0001.zip data-001.zip $DATA_DIR
# download_and_unzip https://huggingface.co/datasets/animelover/genshin-impact-images/resolve/main/data/data-0002.zip data-002.zip $DATA_DIR
# download_and_unzip https://huggingface.co/datasets/animelover/genshin-impact-images/resolve/main/data/data-0003.zip data-003.zip $DATA_DIR
# download_and_unzip https://huggingface.co/datasets/animelover/genshin-impact-images/resolve/main/data/data-0004.zip data-004.zip $DATA_DIR
# download_and_unzip https://huggingface.co/datasets/animelover/genshin-impact-images/resolve/main/data/data-0005.zip data-005.zip $DATA_DIR

remove_nsfw_images
