def add_v2_memory_flags(parser):
    parser.add_argument("--single_backbone", action="store_true")
    parser.add_argument("--flat_memory", action="store_true")
    parser.add_argument("--train_compression", action="store_true")
    parser.add_argument("--compression_epochs", type=int, default=50)
    parser.add_argument("--codec_path", default=None, help="Load or save codec weights.")
    parser.add_argument("--tau_recent", type=int, default=100)
    parser.add_argument("--tau_medium", type=int, default=500)
    parser.add_argument(
        "--age_mode",
        choices=["per_task", "per_example", "per_class"],
        default="per_task",
    )
    parser.add_argument("--log_memory", action="store_true")
    parser.add_argument(
        "--max_bytes",
        type=int,
        default=None,
        help="Evict oldest stored codes once this byte budget is exceeded.",
    )


def build_v2_memory(args, device, memory_cls):
    return memory_cls(
        device=device,
        tau_recent=getattr(args, "tau_recent", 100),
        tau_medium=getattr(args, "tau_medium", 500),
        flat_memory=getattr(args, "flat_memory", False),
        max_bytes=getattr(args, "max_bytes", None),
    )


def age_after_add(memory, args, scope):
    """scope is 'example', 'class', or 'task'."""
    mode = getattr(args, "age_mode", "per_task")
    if mode == "per_example" and scope == "example":
        memory.increment_episode()
    elif mode == "per_class" and scope == "class":
        memory.increment_episode()
    elif mode == "per_task" and scope == "task":
        memory.increment_episode()
