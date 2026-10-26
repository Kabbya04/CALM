def add_common_flags(parser, imagenet=True):
    if imagenet:
        parser.add_argument(
            "--imagenet_root",
            required=True,
            help="Existing ImageNet root (val/<wnid>/*.JPEG). Do not rearrange it.",
        )
        parser.add_argument("--num_tasks", type=int, default=10)
        parser.add_argument("--k_shot", type=int, default=5)
        parser.add_argument("--split", default="val")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser
