from runners.kge_runner import train_gnn_kge_model
from src.config import get_default_arguments


def main():
    args = get_default_arguments()
    train_gnn_kge_model(args)


if __name__ == "__main__":
    main()
