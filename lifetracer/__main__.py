import argparse
import warnings
import lifetracer

def main():
    modules = {
        "extract_heatmap": lifetracer.extract_heatmap, # Module 1
        "find_peaks": lifetracer.find_peaks, # Module 2
        "plot_heatmap": lifetracer.plot_heatmap,
    }

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"lifetracer {lifetracer.__version__}",
    )

    parser.add_argument(
        "--module",
        type=str,
        required=True,
        help="select a module from the following: [ {} ]".format(", ".join(modules.keys())),
        choices=modules.keys(),
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file.",
    )

    # after selecting the module, show the arguments for that module



    # Suppress some warnings
    warnings.filterwarnings("ignore", message="^.* socket cannot be initialized.*$")

    try:
        args = parser.parse_args()
        modules[args.module].main(args)
    except TypeError:
        parser.print_help()

    

    # subparsers = parser.add_subparsers(title="Choose a module", required=True)
    # subparsers.required = True

    # for key in modules:
    #     module_parser = subparsers.add_parser(
    #         key,
    #         description=modules[key].__doc__,
    #         formatter_class=argparse.RawTextHelpFormatter,
    #     )
    #     modules[key].add_args(module_parser)
    #     module_parser.set_defaults(func=modules[key].main)

    # try:
    #     args = parser.parse_args()
    #     args.func(args)
    # except TypeError:
    #     parser.print_help()
    
    


if __name__ == "__main__":
    main()
