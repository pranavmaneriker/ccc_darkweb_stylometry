ORIG_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CUDA_AVAILABLE_DEVICES="0,1"
export CUDA_AVAILABLE_DEVICES

# cd $SCRIPT_DIR/.. 
#for model in "dread_base" "thehub_base" "reddit_2018_sample_base" "reddit_2019_sample_base"; do
for model in "reddit_2018_sample_base" "reddit_2019_sample_base"; do
    for tok in 32 64; do
        for ds in "thehub" "dread" "reddit_2018_sample" "reddit_2019_sample" "dread"; do
            echo "Model": $model
            echo "Dataset:" $ds;
            author_key="creator_id"
            body_key="content"
            if [[ $ds == *"reddit"* ]]; then
                author_key="author"
                body_key="body"
            fi
            model_name=$(ls $SCRIPT_DIR/../../train/LUAR-LLNL/output/$model/lightning_logs/version_0/checkpoints/)
            mkdir -p $SCRIPT_DIR/../results/split/$model
            python -u $SCRIPT_DIR/../run_eval.py --dataset_dir $SCRIPT_DIR/../../data/final/$ds --output_file $SCRIPT_DIR/../results/split/$model/"$ds"_"$tok".json --queries test_queries.jsonl --targets test_targets.jsonl \
            --tokenizer_path $SCRIPT_DIR/../../train/LUAR-LLNL/scripts/pretrained_weights/ --checkpoint_path $SCRIPT_DIR/../../train/LUAR-LLNL/output/$model/lightning_logs/version_0/checkpoints/$model_name \
                --author_key $author_key --text_key $body_key --use_cuda --tokens_per_comment $tok --use_dp --batch_size 16 --from_lightning
        done;
    done
done

cd $ORIG_DIR