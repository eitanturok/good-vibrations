# Literature review

## Sections

1. **Imaging beyond cameras**: wave imaging vs response inversion
2. **Sensing inside enclosures / hidden scenes**
3. **Inference from vibration**: passive vs active
4. **Optical vibration sensing**: the hardware lineage
5. **Learning to map non-visual signals → images**: cross-modal supervision, learned inverse problems (core section; this is an ML paper)

## Columns

- **Regime**: *Wave* = time-of-flight/echoes, resolution set by wavelength · *Response* = excite a fixed system, the object changes its FRF/modes/RIR · *Source* = the object emits (impact) · *Recovery* = the goal is the vibration signal itself · *Semantic* = audio → plausible image, no geometry
- **Excitation**: active / passive or ambient / impact
- **Output**: signal → class → scalar or location → image or geometry
- **Learning**: none / classical ML / deep / deep + physics
- `?` = not verified from the paper

## Print & read (one per area)

1. [Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry](https://openaccess.thecvf.com/content/ICCV2025/papers/Kichler_Learning_to_See_Inside_Opaque_Liquid_Containers_using_Speckle_Vibrometry_ICCV_2025_paper.pdf) (ICCV'25): nearest neighbour, same sensor
2. [The Visual Microphone](https://people.csail.mit.edu/mrub/papers/VisualMic_SIGGRAPH2014.pdf) (SIGGRAPH'14): optical vibration sensing
3. [Shape and Material from Sound](https://papers.NeurIPS.cc/paper_files/paper/2017/file/f4552671f8909587cf485ea990207f3b-Paper.pdf) (NeurIPS'17): physics + learning inference from vibration
4. [Through-Wall Human Pose Estimation Using Radio Signals](https://people.csail.mit.edu/mingmin/papers/rfpose-cvpr-zhao.pdf) (CVPR'18): cross-modal supervision, non-visual signal → image
5. [Acoustic Non-Line-of-Sight Imaging](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lindell_Acoustic_Non-Line-Of-Sight_Imaging_CVPR_2019_paper.pdf) (CVPR'19): physics-based wave imaging of hidden scenes

## Papers

| # | Paper | Signal / sensor | Regime | Excitation | Output | Learning | Section (main / also) |
|---|---|---|---|---|---|---|---|
| 1 | **Ours** | Structural vib. / speckle grid, 2500 fps | Response | Active chirp | **Image / mask** | Deep, camera-supervised | 5 / 3, 4 |
| 2 | [The Boombox: Visual Reconstruction from Acoustic Vibrations](https://arxiv.org/abs/2105.08052) (Chen+ CoRL'21) | Structural vib. / contact mics | Source | Impact | **Image** | Deep, camera-supervised | 5 / 3 |
| 3 | [Learning to See Inside Opaque Liquid Containers using Speckle Vibrometry](https://openaccess.thecvf.com/content/ICCV2025/html/Kichler_Learning_to_See_Inside_Opaque_Liquid_Containers_using_Speckle_Vibrometry_ICCV_2025_paper.html) (Kichler+ ICCV'25) | Structural vib. / speckle grid | Response | Active + ambient | Class (fill level) | Deep (Vibration Transformer) | 2 / 3, 4 |
| 4 | [Wave-Former: Through-Occlusion 3D Reconstruction](https://arxiv.org/abs/2511.14152) (Dodds+ arXiv'25) | mmWave radar | Wave | Active | **3D point cloud** | Deep (PoinTr transformer), synthetic training | 2 / 5 |
| 5 | [Non-Line-of-Sight 3D Object Reconstruction via mmWave Surface Normal Estimation](https://news.mit.edu/2025/new-imaging-technique-reconstructs-hidden-object-shapes-0701) (mmNorm, Dodds+ MobiSys'25) | mmWave radar | Wave | Active | **3D geometry** | ? | 2 / 1 |
| 6 | [Seeing through boxes: Non-Line-of-Sight 3D Reconstruction from Radar Signals](https://sens.epfl.ch/research/geraf/) (GeRaF 2.0, Lu+ CVPR'26) | mmWave radar | Wave | Active | **3D mesh (SDF)** | Deep + physics (neural field) | 2 / 5 |
| 7 | [BatVision: Learning to See 3D Spatial Layout with Two Ears](https://arxiv.org/abs/1912.07011) (Christensen+ ICRA'20) | Airborne sound / binaural mics | Wave | Active chirp | **Image** (depth, grayscale) | Deep, camera-supervised | 5 / 1 |
| 8 | [VisualEchoes: Spatial Image Representation Learning through Echolocation](https://arxiv.org/abs/2005.01616) (Gao+ ECCV'20) | Airborne echoes (sim) | Wave | Active | Representation → depth, normals | Deep, self-supervised | 5 / 1 |
| 9 | [Through-Wall Human Pose Estimation Using Radio Signals](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhao_Through-Wall_Human_Pose_CVPR_2018_paper.html) (RF-Pose, Zhao+ CVPR'18) | RF / antennas | Wave | Active | **Image-space** 2D pose | Deep, camera-supervised | 5 / 2 |
| 10 | [SoundCam: A Dataset for Finding Humans Using Room Acoustics](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a4289154c9209b679ac761a50d5fec3a-Abstract.html) (Wang+ NeurIPS'23) | Room acoustics / mic array | Response | Active | Location, identity | kNN, linear, deep | 1 / 5 |
| 11 | [Acoustic Non-Line-of-Sight Imaging](https://openaccess.thecvf.com/content_CVPR_2019/html/Lindell_Acoustic_Non-Line-Of-Sight_Imaging_CVPR_2019_paper.html) (Lindell+ CVPR'19) | Airborne sound / mic array | Wave | Active chirp | 3D geometry | None (migration) | 2 / 1 |
| 12 | [Shape and Material from Sound](https://papers.neurips.cc/paper_files/paper/2017/hash/f4552671f8909587cf485ea990207f3b-Abstract.html) (Zhang+ NeurIPS'17) | Impact sound / mic | Source | Impact | Shape, material | Deep + physics (analysis-by-synthesis) | 3 / 5 |
| 13 | [The ObjectFolder Benchmark: Multisensory Learning with Neural and Real Objects](https://openaccess.thecvf.com/content/CVPR2023/html/Gao_The_ObjectFolder_Benchmark_Multisensory_Learning_With_Neural_and_Real_Objects_CVPR_2023_paper.html) (Gao+ CVPR'23) | Impact audio + vision + touch | Source | Impact | Class, contact loc., shape | Deep | 3 |
| 14 | [RealImpact: A Dataset of Impact Sound Fields for Real Objects](https://openaccess.thecvf.com/content/CVPR2023/html/Clarke_RealImpact_A_Dataset_of_Impact_Sound_Fields_for_Real_Objects_CVPR_2023_paper.html) (Clarke+ CVPR'23) | Impact sound / mics | Source | Impact | Dataset; listener-location class | Deep baselines | 3 |
| 15 | [SonicSense: Object Perception from In-Hand Acoustic Vibration](https://arxiv.org/abs/2406.17932) (2024) | In-hand vib. / contact mics | Source | Robot interaction | Class, count, shape | Deep | 3 |
| 16 | [Non-Contact Vibro-Acoustic Object Recognition Using Laser Doppler Vibrometry and CNNs](https://doi.org/10.3390/s22239360) (Sensors'22) | Structural vib. / LDV | Response? | ? | Class | Deep | 3 / 4 |
| 17 | [VibSense: Sensing Touches on Ubiquitous Surfaces through Vibration](https://personal.stevens.edu/~ychen6/papers/VibSense%20Sensing%20Touches%20on%20Ubiquitous%20Surfaces%20through%20Vibration.pdf) (Liu+) | Surface vib. / 1 sensor | Response (active mode) | Active | Touch location, object ID | Classical ML | 3 |
| 18 | [SAWSense: Using Surface Acoustic Waves for Surface-bound Event Recognition](https://eecsnews.engin.umich.edu/sensor-enables-high-fidelity-input-from-everyday-objects-human-body/) (CHI'23; press link) | Surface acoustic waves / VPU | Source | Touch | Event class | ML | 3 |
| 19 | [Vibro-Sense: Robust Vibration-based Impulse Response Localization and Trajectory Tracking for Robotic Hands](https://arxiv.org/abs/2601.20555) (2026) | Vib. / piezo mics on hand | Source | Contact impulse | Location (<5 mm) | Deep (AST) | 3 |
| 20 | [Identification of member connectivity and mass changes on a two-storey framed structure using FRFs and ANNs](https://sciencedirect.com/science/article/abs/pii/S0022460X13001491) (JSV'13) | Structural FRF | Response | Active | **Mass location** | NN + PCA | 3 / 1 |
| 21 | [The location of a concentrated mass on rectangular plates from measurements of natural vibrations](https://www.researchgate.net/publication/223375326_The_location_of_a_concentrated_mass_on_rectangular_plates_from_measurements_of_natural_vibrations) | Natural modes | Response | Active | Mass location | None (inverse) | 3 / 1 |
| 22 | [Damage Localization in Composite Plates Using Wavelet Transform and 2-D CNNs](https://doi.org/10.3390/s21175825) (Sensors'21) | Structural vib. | Response | Active | Damage location | Deep | 3 |
| 23 | [Visual Vibrometry: Estimating Material Properties from Small Motion in Video](https://openaccess.thecvf.com/content_cvpr_2015/html/Davis_Visual_Vibrometry_Estimating_2015_CVPR_paper.html) (Davis+ CVPR'15) | Video | Response (modes) | Ambient | Material properties | Physics + regression | 4 / 3 |
| 24 | [The Visual Microphone: Passive Recovery of Sound from Video](https://people.csail.mit.edu/mrub/VisualMic) (Davis+ SIGGRAPH'14) | Video | Recovery | Passive sound | Audio | None | 4 |
| 25 | [Simultaneous remote extraction of multiple speech sources and heart beats from secondary speckles pattern](https://www.uv.es/~gpoei/articulos/(2009)%20OEx%2017%2021566%20Simultaneous%20remote%20extraction%20of%20multiple%20speech%20sources%20and%20heart%20beats%20from%20secondary%20speckles%20pattern) (Zalevsky+ Opt. Express'09) | Speckle / defocused camera | Recovery | Passive | Audio, heartbeat | None | 4 |
| 26 | [Dual-Shutter Optical Vibration Sensing](https://openaccess.thecvf.com/content/CVPR2022/html/Sheinin_Dual-Shutter_Optical_Vibration_Sensing_CVPR_2022_paper.html) (Sheinin+ CVPR'22) | Speckle / 2 cameras | Recovery | Passive | Audio, modes | None | 4 |
| 27 | [Event2Audio: Event-Based Optical Vibration Sensing](https://arxiv.org/abs/2507.03273) (Cai+ ICCP'25) | Speckle / event camera | Recovery | Passive | Audio | ? | 4 |
| 28 | [AudioToken: Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation](https://arxiv.org/abs/2305.13050) (Interspeech'23) | Audio clip | Semantic | — | Image (semantic) | Deep (diffusion) | 5 |
| 29 | [SonicDiffusion: Audio-Driven Image Generation and Editing with Pretrained Diffusion Models](https://arxiv.org/abs/2405.00878) (2024) | Audio clip | Semantic | — | Image (semantic) | Deep (diffusion) | 5 |
| 30 | [Estimating the Material Properties of Fabric from Video](https://openaccess.thecvf.com/content_iccv_2013/html/Bouman_Estimating_the_Material_2013_ICCV_paper.html) (Bouman+ ICCV'13) | Video | Response | Ambient (wind) | Stiffness, density | Regression | 4 / 3 |
| 31 | [Visual Vibration Tomography: Estimating Interior Material Properties from Monocular Video](https://openaccess.thecvf.com/content/CVPR2022/html/Feng_Visual_Vibration_Tomography_Estimating_Interior_Material_Properties_From_Monocular_Video_CVPR_2022_paper.html) (Feng, Ogren+ CVPR'22) | High-speed video | Response (modes) | Active | **Interior property map** | Physics (inverse) | 4 / 3 |
| 32 | [Visual Surface Wave Elastography: Revealing Subsurface Physical Properties via Visible Surface Waves](https://openaccess.thecvf.com/content/ICCV2025/papers/Ogren_Visual_Surface_Wave_Elastography_Revealing_Subsurface_Physical_Properties_via_Visible_ICCV_2025_paper.pdf) (Ogren+ ICCV'25) | Video | Wave (surface) | Active | Thickness, stiffness | Physics (optimization) | 4 / 3 |
| 33 | [Hearing the Room Through the Shape of the Drum: Modal-Guided Sound Recovery from Multi-Point Surface Vibrations](https://arxiv.org/abs/2604.26678) (Bagon, Kichler, Sheinin CVPR'26) | Speckle / multi-point | Recovery | Passive | Audio | Physics (modal model) | 4 |
| 34 | [High-throughput anomaly detection in document envelopes with 3D millimeter-wave imaging](https://spiedigitallibrary.org/conference-proceedings-of-spie/11745/117450G/High-throughput-anomaly-detection-in-document-envelopes-with-3D-millimeter/10.1117/12.2587522.full) (SPIE'21) | mmWave, 24 GHz | Wave | Active | Image + anomaly flag | Logistic regression | 2 / 1 |
| 35 | [Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging](https://www.nature.com/articles/ncomms1747) (Velten+ Nat. Commun.'12) | Pulsed laser / streak camera | Wave | Active | 3D geometry | None | 2 / 1 |
| 36 | [Analyzing Physical Impacts Using Transient Surface Wave Imaging](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Analyzing_Physical_Impacts_Using_Transient_Surface_Wave_Imaging_CVPR_2023_paper.html) (Zhang, Sheinin+ CVPR'23) | Speckle / dual-shutter | Wave (surface) | Impact | Impact location, material cues | Physics | 4 / 3 |
| 37 | [See Through Walls with WiFi!](https://doi.org/10.1145/2486001.2486039) (Adib & Katabi SIGCOMM'13) | WiFi | Wave | Active | Motion, count | None | 1 / 2 |
| 38 | [Person-in-WiFi: Fine-Grained Person Perception Using WiFi](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Person-in-WiFi_Fine-Grained_Person_Perception_Using_WiFi_ICCV_2019_paper.html) (Wang+ ICCV'19) | WiFi | Wave | Active | **Image-space** mask + pose | Deep, camera-supervised | 5 / 1 |
| 39 | [DensePose From WiFi](https://arxiv.org/abs/2301.00250) (Geng+ arXiv'22) | WiFi | Wave | Active | **Image-space** dense UV | Deep, camera-supervised | 5 / 1 |
| — | *TODO: EIT / diffuse optical tomography with learned priors* | Boundary measurements | Response | Active | Image | Deep | 1 / 5 |
