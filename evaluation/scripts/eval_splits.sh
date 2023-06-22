ORIG_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CUDA_AVAILABLE_DEVICES="0,1"
export CUDA_AVAILABLE_DEVICES

# cd $SCRIPT_DIR/.. 
for tok in 32 64; do
    for ds in "thehub" "dread" "reddit_2018_sample" "reddit_2019_sample" "dread"; do
        echo "Dataset:" $ds;
        author_key="creator_id"
        body_key="content"
        if [[ $ds == *"reddit"* ]]; then
            author_key="author"
            body_key="body"
        fi
        python -u $SCRIPT_DIR/../run_eval.py --dataset_dir $SCRIPT_DIR/../../data/final/$ds --output_file $SCRIPT_DIR/../results/split/scale_mud/"$ds"_"$tok".json --queries test_queries.jsonl --targets test_targets.jsonl \
        --tokenizer_path $SCRIPT_DIR/../../models/tokenizers/ --checkpoint_path $SCRIPT_DIR/../../models/scale-luar/scale_mud.pt \
            --author_key $author_key --text_key $body_key --use_cuda --tokens_per_comment $tok --use_dp --batch_size 16
    done;
done

cd $ORIG_DIR