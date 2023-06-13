from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir")
    parser.add_argument("--ndjson_file")
    parser.add_argument("--output_format", choices=["parquet"])