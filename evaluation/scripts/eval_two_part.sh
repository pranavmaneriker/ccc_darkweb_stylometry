ORIG_DIR=$(pwd)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CUDA_AVAILABLE_DEVICES="0,1"
export CUDA_AVAILABLE_DEVICES

# cd $SCRIPT_DIR/.. 
for ds in $(ls ../data/final/two_part); do
    echo "Dataset:" $ds;
    author_key="creator"
    body_key="content"
    if [[ $ds == *"reddit"* ]]; then
        author_key="author"
        body_key="body"
    fi
    python -u $SCRIPT_DIR/../run_eval.py --dataset_dir ../data/final/two_part/$ds --output_file results/two_part_$ds.json --queries queries.jsonl --targets targets.jsonl \
         --author_key $author_key --text_key $body_key --use_cuda --tokens_per_comment 32 --use_dp --batch_size 16 
done;

cd $ORIG_DIR