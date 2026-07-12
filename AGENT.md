We want to generate an image of what is inside of a closed box from it's vibrations. 

We collect our own dataset and train a deep learning model to generate this image.

# The Data

1. We place an object, e.g. a cube, in a box.
2. We take a photo of the the object in the box from a bird's eye view.
3. We shine a `l_h x l_w` grid of laser points onto the side of the box. 
4. We play an audio file, e.g. a chrip, from one of `S` speakers. This vibrates the box but the vibrations are tiny and not visible to the human eye.
5. As we play the audio, we record a video of the laser speckles vibrating.

Our raw dataset is `(laser_vibrations, overhead_image)`.

To process this data
1. From the `overhead_image` we create a segementation mask of then object.
2. From the `laser_vibrations` we recover an fft signal.

Our processed dataset is (`fft`, `smask`).

We then train a model to predict `smask` given `fft`.

Our data is stored as

```
experiment_dir/
    audio/
        50_1000_3sec/
            audio.mp3
            samples.jsonl
    images/
        000000/
            00_raw.png
            01_cropped.png
            02_smask.png
            03_smask.npy
            04_overhead_with_smask.png
            samples.jsonl
    samples/
        000000/
            laser/
                00_roi_rows.png
                01_roi_cols.png
                02_roi.png
                03_speckles.png
            image/
                00_raw.png
                01_cropped.png
                02_smask.png
                03_smask.npy
                04_overhead_with_smask.png
                05_overhead_with_speaker.png
            vibration/
                00_raw_vibrations.npy
                01_raw_shifts.npy
                02_clean_shifts.npy
                03_fft.npz
                04_recovered_audio.mp3
            audio.mp3
            recovered_audio.mp3
            overhead.png
            times.jsonl
            metadata.jsonl
```

```
dataset_dir/
    dataset-1/
        samples/
            000000/
                laser/
                    00_roi_rows.png
                    01_roi_cols.png
                    02_roi_grid.png
                    03_speckles.png
                image/
                    00_raw.png
                    01_cropped.png
                    02_smask.png
                    03_smask.npy
                    04_overhead_with_smask.png
                    05_overhead_with_speaker.png
                    06_downsampled_smask.png
                    07_downsampled_smask.npy
                vibration/
                    00_raw_vibrations.npy
                    01_raw_shifts.npy
                    02_clean_shifts.npy
                    03_fft.npz
                    04_recovered_audio.wav
                    05_signaled_fft.npy
                    06_normalized_fft.npy
                    07_tokenized_fft.npy
                audio.mp3
                recovered_audio.mp3
                overhead.png
                metadata.jsonl
                times.jsonl
                X.npy
                Y.npy
        mds/ 			
            metadata.jsonl
            index.jsonl
            shards_00.jsonl
            shards_01.jsonl
            shards_02.jsonl
```

# Literature
Previosuly, my lab predicted how much liquid is in a container from the vibrations `assets\Kichler_Learning_to_See_Inside_Opaque_Liquid_Containers_using_Speckle_Vibrometry_ICCV_2025_paper.pdf`. This was a classification problem. We are now trying to do a generation problem. We use the same physical setup from the experiment.