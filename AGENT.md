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

To see how the data is strucutred, read `data/README.md`.

Logs about the data can be found in `experiment_dir/logs.md` and in `modal app APP-ID logs`. These logs should be the first place you check when there is an issue with data processing.

# Literature
Previosuly, my lab predicted how much liquid is in a container from the vibrations `assets\Kichler_Learning_to_See_Inside_Opaque_Liquid_Containers_using_Speckle_Vibrometry_ICCV_2025_paper.pdf`. This was a classification problem. We are now trying to do a generation problem. We use the same physical setup from the experiment.