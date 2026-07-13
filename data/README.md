The data lives in two differnet directories:
* `experiment_dir` stores the data we recorded and partially processes it
* `dataset_dir` stores the fully processed dataset and formats it as MDS files which we pass to our deep learning model

We seperate these two because the dataset format changes depending on the modal architecture.

The directory structures are
```
experiment_dir/
    audio/
        50_1000_3sec/
            audio.wav
            spectogram.png
            samples.jsonl
    images/
        000000/
            00_raw.png
            01_cropped.png
            02_smask.png
            03_smask.npy
            04_overhead_masked.png     (mask overlay + avg COM crosshair, no boxes/confidence)
            05_overhead_scored.png     (mask + boxes + confidence labels)
            samples.jsonl
            smasks/
                cube1.png
                cube1.npy
                cube2.png
                cube2.npy
                cylinder1.png
                cylinder1.npy
                all.png                (all objects, one solid color each, no overlap/boxes)
                all.npy                (label map: 0=bg, i+1=object i)
                metadata.jsonl         (per-object: name, com, score, box)
    samples/
        000000/
            laser/
                00_roi_rows.png
                01_roi_cols.png
                02_roi.png
                03_speckles.png
            image/
                00_raw.png              (symlink)
                01_cropped.png          (symlink)
                02_smask.png            (symlink)
                03_smask.npy            (symlink)
                04_overhead_masked.png  (symlink)
                05_overhead_scored.png  (symlink)
                06_overhead_speaker.png (adds speaker icon to 04_overhead_masked.png)
                smasks/                 (symlink to images/000000/smasks/)
            vibration/
                00_raw_vibrations.npy
                01_raw_shifts.npy
                02_clean_shifts.npy
                03_fft.npz
                04_recovered_audio.wav
                05_recovered_spectogram.png
            audio.wav
            recovered_audio.wav
            overhead.png             (symlink to image/06_overhead_speaker.png)
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
                    05_recovered_spectogram.png
                    06_signaled_fft.npy
                    07_normalized_fft.npy
                    08_tokenized_fft.npy
                audio.wav
                recovered_audio.wav
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