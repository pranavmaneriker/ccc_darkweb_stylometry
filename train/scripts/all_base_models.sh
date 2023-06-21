# assume this will be sourced in an activated virtualenv

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../LUAR-LLNL/
for ds_name in "dread" "thehub" "reddit_2018_sampled" "reddit_2019_sampled"; do
if []
python -u src/main.py --dataset_name $ds_name --experiment_id "$ds_name"_base --text_key content --do_learn --gpus 2 --token_max_length 64 --batch_size 256 --pin_memory --episode_length 4
