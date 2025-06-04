import argparse


def get_parser(**kwargs):
    add_help = kwargs.get("add_help", True)

    parser = argparse.ArgumentParser(
        description="Perform PE on real events",
        add_help=add_help,
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./outdir/",
        help="Output directory",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        help="ID of the event",
    )
    parser.add_argument(
        "-r",
        "--relative-binning",
        action="store_true",
        help="Enable relative binning",
    )

    return parser
