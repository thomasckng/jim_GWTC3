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
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Relative binning mode: 0=normal, 1=fixed reference, 2=optimised reference",
    )

    return parser
