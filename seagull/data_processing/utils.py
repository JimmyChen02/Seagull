from functools import partial
from typing import List, Dict, Any, Union, Optional, Tuple

import datasets

datasets.disable_caching()


def get_torch_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    dataset.set_format(type="torch")
    return dataset


def _merge_scene_uncanny_caption(
    data_instances: Dict[str, List[Any]],
    scene_colname_and_special_token: Tuple[str, str],
    uncanny_colname_and_special_token: Tuple[str, str],
    caption_colname_and_special_token: Tuple[str, str],
    end_of_caption_special_token: str,
    merge_colname: str,
) -> Dict[str, List[Any]]:
    """
    Adds a list of merged scenes, uncanny descriptions, and captions under the given key to the given dictionary of data
    instances.

    Parameters
    ----------
    data_instances : Dict[str, List[Any]]
        A dictionary of data instances containing a list of scenes, uncanny descriptions, and captions each associated
        with their corresponding column names as keys.
    scene_colname_and_special_token : Tuple[str, str]
        A tuple containing the column name for scenes and the special token indicating the beginning of a scene
        description, respectively.
    uncanny_colname_and_special_token : Tuple[str, str]
        A tuple containing the column name for uncanny descriptions and the special token indicating the beginning of an
        uncanny description, respectively.
    caption_colname_and_special_token : Tuple[str, str]
        A tuple containing the column name for captions and the special token indicating the beginning of a caption,
        respectively.
    end_of_caption_special_token : str
        The special token indicating the end of a caption.
    merge_colname : str
        The column name under which to add the merged data instances.

    Returns
    -------
    Dict[str, List[Any]]
        A dictionary of the given data instances with an added key, ``merge_colname``, associated with a list of the
        merged data instances.
    """
    for colname, special_token in [
        scene_colname_and_special_token,
        uncanny_colname_and_special_token,
        caption_colname_and_special_token,
    ]:
        if merge_colname not in data_instances:
            data_instances[merge_colname] = [""] * len(data_instances[colname])
        end_of_caption = end_of_caption_special_token if colname == caption_colname_and_special_token[0] else ""
        if colname not in data_instances:
            end_of_caption = ""
            data_instances[colname] = [""] * len(data_instances[merge_colname])

        data_instances[merge_colname] = [
            f"{merged_value} {special_token} {col_value} {end_of_caption}".strip()
            for col_value, merged_value in zip(data_instances[colname], data_instances[merge_colname])
        ]
    return data_instances


def generate_newyorker_lm_text_dataset(
    newyorker_dataset: Union[datasets.Dataset, datasets.dataset_dict.DatasetDict],
    scene_colname_and_special_token: Tuple[str, str],
    uncanny_colname_and_special_token: Tuple[str, str],
    caption_colname_and_special_token: Tuple[str, str],
    end_of_caption_special_token: str,
    merge_colname: str = "text",
    batch_size: int = 4000,
    remove_cols: Optional[list] = None,
) -> Union[datasets.Dataset, datasets.dataset_dict.DatasetDict]:
    formatting_fn = partial(
        _merge_scene_uncanny_caption,
        scene_colname_and_special_token=scene_colname_and_special_token,
        uncanny_colname_and_special_token=uncanny_colname_and_special_token,
        caption_colname_and_special_token=caption_colname_and_special_token,
        end_of_caption_special_token=end_of_caption_special_token,
        merge_colname=merge_colname,
    )
    newyorker_dataset = newyorker_dataset.map(formatting_fn, batched=True, batch_size=batch_size).shuffle(seed=4740)
    if remove_cols is not None:
        newyorker_dataset = newyorker_dataset.remove_columns(remove_cols)
    return newyorker_dataset