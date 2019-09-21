import argparse
import pandas as pd
import argcomplete


def create_unified_file(marketing_file: str,
                        non_marketing_file: str,
                        output_file: str) -> None:
    marketing_df = pd.read_csv(marketing_file, delimiter='\t', header=None, names=['id', 'tweet'])
    print(f'No. of Marketing tweets: {len(marketing_df)}')
    marketing_df['label'] = ['Marketing'] * len(marketing_df)

    non_marketing_df = pd.read_csv(non_marketing_file, delimiter='\t', header=None, names=['id', 'tweet'])
    non_marketing_df['label'] = ['Not Marketing'] * len(non_marketing_df)
    print(f'No. of Non Marketing tweets: {len(non_marketing_df)}')

    unified_df = pd.concat([marketing_df, non_marketing_df], ignore_index=True, sort=['id'])
    print(f'No. total labeled tweets: {len(unified_df)}')

    with open(output_file, 'w+') as out_file:
        unified_df.to_csv(out_file, index=False)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--marketing-file", type=str, help='Marketing tweets TXT file', required=True)
    argument_parser.add_argument("--non-marketing-file", type=str, help='Non Marketing tweets TXT file', required=True)
    argument_parser.add_argument("--output-file", type=str, help='Output titles txt', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    create_unified_file(args.marketing_file, args.non_marketing_file, args.output_file)


if __name__ == '__main__':
    main()
