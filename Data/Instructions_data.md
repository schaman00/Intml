# Instructions for Downloading and Preparing the Dataset

This project uses the **Multi-dSprites Colored on Colored** dataset. To download and prepare the data, first open a terminal and navigate to the project directory containing the `prepare_data.py` script:

```bash
cd /path/to/your/project
```

Then download the dataset using the following command:

```bash
wget https://storage.googleapis.com/multi-object-datasets/multi_dsprites/multi_dsprites_colored_on_colored.tfrecords
```

After downloading the file, make sure that it is located in the same directory as `prepare_data.py`. The project structure should be:

```text
.
├── prepare_data.py
└── multi_dsprites_colored_on_colored.tfrecords
```

Finally, run the data preparation script:

```bash
python prepare_data.py
```

The script will read the downloaded TFRecord file and generate the processed data files required by the project.
