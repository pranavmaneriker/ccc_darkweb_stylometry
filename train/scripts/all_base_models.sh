# assume this will be sourced in an activated virtualenv

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../LUAR-LLNL/
#for ds_name in "dread" "thehub" "reddit_2018_sample" "reddit_2019_sample"; do
for ds_name in "reddit_2018_sample" "reddit_2019_sample"; do
    echo "Dataset:" $ds_name
    author_key="creator_id"
    body_key="content"
    if [[ $ds_name == *"reddit"* ]]; then
        author_key="author"
        body_key="body"
    fi
    python -u src/main.py --dataset_name $ds_name --experiment_id "$ds_name"_base --text_key $body_key --do_learn --gpus 2 --token_max_length 64 --batch_size 128 --pin_memory --episode_length 4 &> output/logs/"$ds_name"_base.log
done;