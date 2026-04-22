import json
import os
import shutil
from argparse import ArgumentParser
from typing import Dict, Optional

import datasets
import torch
import yaml
from rich.progress import track

from seagull.data_processing.bbpe import BBPETokenizer
from seagull.data_processing.constants import END_OF_CAPTION_TOKEN
from seagull.model.heads.seagull_lm import SeagullLM
from seagull.utils.torch_utils import set_pytorch_backends, set_seed, get_device
from seagull.utils.utils import colored

set_pytorch_backends()



def _append_net_ids_to_file(filename: str, net_ids: str) -> None:
    with open(filename, "r+") as fp:
        content = fp.read().splitlines(True)
        fp.seek(0, 0)

        start_line = "# AUTO-GENERATED (DO NOT MODIFY)\n"
        net_ids = f"# NET IDS: {net_ids.upper()}\n\n"

        if content[0] == start_line:
            content = content[3:]
        fp.writelines([start_line, net_ids] + content)


def _write_seagull_files(net_ids: Optional[str], seagull_outputs_path: str, is_milestone_submission: bool) -> None:
    seagull_basepath = "seagull"
    files_to_copy = [
        os.path.join(seagull_basepath, "data_processing/utils.py"),
        os.path.join(seagull_basepath, "data_processing/sequence_sampler.py"),
        os.path.join(seagull_basepath, "model/components/embedding.py"),
        os.path.join(seagull_basepath, "nn/transformer/mha.py"),
    ]
    if not is_milestone_submission:
        files_to_copy = files_to_copy + [
            os.path.join(seagull_basepath, "nn/transformer/ffn.py"),
            os.path.join(seagull_basepath, "model/components/transformer_layer.py"),
            os.path.join(seagull_basepath, "model/heads/seagull_lm.py"),
            os.path.join(seagull_basepath, "model/heads/lm_utils.py"),
        ]
    for file in files_to_copy:
        shutil.copy2(file, seagull_outputs_path)
        if net_ids is not None:
            _append_net_ids_to_file(
                filename=os.path.join(seagull_outputs_path, os.path.basename(file)), net_ids=net_ids
            )


def _make_output_dirs(
    basepath_to_store_submission: str, is_milestone_submission: bool
) -> Dict[str, str]:
    seagull_outputs_path = os.path.join(basepath_to_store_submission, "seagull")

    os.makedirs(basepath_to_store_submission, exist_ok=True)
    if is_milestone_submission:
        os.makedirs(seagull_outputs_path, exist_ok=True)
    if not is_milestone_submission:
        os.makedirs(seagull_outputs_path, exist_ok=True)

    return {
        "basepath_to_store_submission": basepath_to_store_submission,
        "seagull_outputs_path": seagull_outputs_path,
    }


def _delete_if_exists(filename: str) -> None:
    try:
        os.remove(filename)
    except OSError:
        pass


def main(
    basepath_to_store_submission: str,
    is_milestone_submission: bool = False,
    net_ids: Optional[str] = None,
) -> None:
    set_seed(4740)
    if basepath_to_store_submission.endswith("/"):
        basepath_to_store_submission = basepath_to_store_submission[:-1]

    if is_milestone_submission:
        basepath_to_store_submission = os.path.join(basepath_to_store_submission, "milestone_submission")
    else:
        basepath_to_store_submission = os.path.join(basepath_to_store_submission, "hw3_submission")
    
    _delete_if_exists(f"{basepath_to_store_submission}.zip")
    all_output_paths = _make_output_dirs(
        basepath_to_store_submission,
        is_milestone_submission=is_milestone_submission,
    )

    if net_ids is None:
        raise ValueError("must include '--net-ids' as a comma-separated string (e.g., '<net-id-1>,<net-id-2>')")
    _write_seagull_files(
        seagull_outputs_path=all_output_paths["seagull_outputs_path"],
        is_milestone_submission=is_milestone_submission,
        net_ids=net_ids,
    )

    shutil.make_archive(basepath_to_store_submission, "zip", basepath_to_store_submission)
    shutil.rmtree(basepath_to_store_submission)
    print(f"submission stored at: {basepath_to_store_submission}.zip")
    

def argparser():
    parser = ArgumentParser(description="Make a submission folder for the assignment.")
    parser.add_argument(
        "--basepath-to-store-submission",
        type=str,
        help="The basepath to store all the files required to make a gradescope submission.",
        default=os.getcwd(),
        required=True,
    )
    parser.add_argument(
        "--milestone-submission",
        action="store_true",
        help="Flag to indicate if the current submission is for the milestone.",
    )
    parser.add_argument(
        "--net-ids",
        type=str,
        help="Student net-IDs as a comma-separated string (e.g., '<net-id-1>,<net-id-2>').",
        required=False,
    )
    return parser


if __name__ == "__main__":
    args = argparser().parse_args()
    main(
        basepath_to_store_submission=args.basepath_to_store_submission,
        is_milestone_submission=args.milestone_submission,
        net_ids=args.net_ids,
    )
