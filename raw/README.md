# Raw datasets

Place original dataset files under a task subdirectory, such as `raw/white_background/`. Keep these files unchanged. The runner separately saves the images returned by the loader under `outputs/{task_name}/{generation_version}/inputs/`.

Set `dataset.raw_path` in YAML to the subdirectory name. The loader receives its absolute path as `config["raw_dir"]`.
