import json
import pandas as pd
import sys

def stratify_hotpotqa_dataset(hotpotqa_dev_file, stratified_hotpotqa_output_file):
    with open(hotpotqa_dev_file, encoding='utf-8') as f:
        hotpotqa_full_dev = json.load(f)

    hotpotqa_full_dev_df = pd.DataFrame(hotpotqa_full_dev)

    sample_size = 501
    stratified_hotpotqa_500sample = hotpotqa_full_dev_df.groupby(by=['type'], group_keys=False).apply(lambda x: x.sample(n=int(sample_size * (len(x) / len(hotpotqa_full_dev_df))),random_state=1))

    stratified_hotpotqa_500sample.reset_index(drop=True, inplace=True)

    stratified_hotpotqa_500sample.to_json(stratified_hotpotqa_output_file, orient='records', lines=False)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python stratify_hotpotqa.py <hotpot_dev_distractor_v1.json> <stratified_hotpotqa_500sample.json>")
        sys.exit(1)
    stratify_hotpotqa_dataset(sys.argv[1], sys.argv[2])