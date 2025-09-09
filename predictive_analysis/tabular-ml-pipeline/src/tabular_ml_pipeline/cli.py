import argparse

def main():
    parser = argparse.ArgumentParser(description="Tabular ML Pipeline CLI")
    parser.add_argument("command", choices=["etl", "train", "eval"])
    parser.add_argument("--dataset", choices=["telco", "credit_risk"], required=True)
    args = parser.parse_args()
    print(f"Running {args.command} on {args.dataset} dataset...")

if __name__ == "__main__":
    main()
