# Face Recognition Model Data Download Instructions

This project uses the **FairFace** and **UTKFace** datasets for training and evaluation. Follow the steps below to download these datasets from Kaggle.

## Prerequisites

- A [Kaggle](https://www.kaggle.com/) account
- [Kaggle API](https://github.com/Kaggle/kaggle-api) installed and configured (`kaggle.json` file in `~/.kaggle/`)

## Download FairFace Dataset

Visit the [FairFace dataset page](https://www.kaggle.com/datasets/aibloy/fairface).


## Download UTKFace Dataset

1. Visit the [UTKFace dataset page](https://www.kaggle.com/datasets/jangedoo/utkface-new).
2. Download manually, or use the Kaggle API:

    ```bash
    kaggle datasets download -d jangedoo/utkface-new
    unzip utkface-new.zip -d ./utkface
    ```

## Notes

- Place the extracted folders inside the `data/` directory.
- Ensure you have sufficient disk space for both datasets.

For more details, refer to the official Kaggle dataset pages.