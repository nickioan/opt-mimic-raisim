import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--name", required=True, type=str, default="trot backward fast", help="Input the name of the trajectory, default is trot forward fast."
        )
    args = parser.parse_args()
    return args

