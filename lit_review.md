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
| 40 | [An Investigation of Acoustic-to-Seismic Coupling to Detect Buried Antitank Landmines](https://ieeexplore.ieee.org/document/927429) (Sabatier & Xiang, IEEE TGRS'01) | Ground vib. / scanning LDV | Response | Active (airborne sound, ~80–300 Hz) | **Vibration image** → mine location | None | 3 / 1, 4 |
| 41 | [Vibro-acoustography: An Imaging Modality Based on Ultrasound-Stimulated Acoustic Emission](https://pmc.ncbi.nlm.nih.gov/articles/PMC21961) (Fatemi & Greenleaf, PNAS'99; Science'98 original) | Hydrophone, after ultrasound radiation force | Response | Active (localized force) | **Image** | None (scanned) | 1 / 3 |
| 42 | [Unsupervised Learning for Industrial Defect Detection: A Case Study on Shearographic Data](https://arxiv.org/abs/2511.02541) (Plassmann+ arXiv'25) | Speckle shearography | Response (loaded) | Active (thermal/pressure load) | Defect map | Deep (autoencoders) | 4 / 3 |
| 43 | [Damage Identification and Health Monitoring of Structural and Mechanical Systems from Changes in Their Vibration Characteristics](https://public.lanl.gov/prime/doebling_svd.pdf) (Doebling, Farrar+ LANL'96) | Structural modes (review) | Response | Active / ambient | Detect → locate → size | Review (model-based) | 3 |
| 44 | [A Review of Vibration-Based Damage Detection in Civil Structures: From Traditional Methods to ML and DL](https://arxiv.org/abs/2004.04373) (Avci+ MSSP'21) | Structural vib. (review) | Response | Active / ambient | Damage class / location | Review (ML, DL) | 3 |
| 45 | Can One Hear the Shape of a Drum? (Kac, Amer. Math. Monthly'66) + [One Cannot Hear the Shape of a Drum](https://arxiv.org/abs/math/9207215) (Gordon, Webb, Wolpert, Bull. AMS'92) | Eigenvalue spectrum (theory) | Response | — | Shape identifiability | Theory | 3 / 1 |
| 46 | [Touch & Activate: Adding Interactivity to Existing Objects using Active Acoustic Sensing](https://www.iplab.cs.tsukuba.ac.jp/en/paper/6151b7ff551fd46e82abdb1b) (Ono+ UIST'13) | Contact speaker + piezo mic | Response | Active sweep | Touch / grasp class | Classical ML (SVM) | 3 |
| 47 | [Knocker: Vibroacoustic-based Object Recognition with Smartphones](https://nmsl.kaist.ac.kr/projects/knocker) (Gong+ IMWUT'19) | Phone mic + IMU | Source | Impact (knock) | Object class | Classical ML | 3 |
| 48 | [Vibrosight: Long-Range Vibrometry for Smart Environment Sensing](https://www.figlab.com/research/2018/vibrosight) (Zhang, Laput, Harrison UIST'18) | Laser vibrometer + retroreflective tags | Source | Passive (appliance) | Activity class | Classical ML | 4 / 3 |
| 49 | [VibroSense: Recognizing Home Activities by Deep Learning Subtle Vibrations … Using Laser Doppler Vibrometry](https://dl.acm.org/doi/10.1145/3411828) (IMWUT'20) | LDV, single point | Source | Passive | Activity class | Deep | 4 / 3 |
| 50 | [LiquID: A Wireless Liquid IDentifier](https://faculty.cc.gatech.edu/~dhekne/liquid_mobisys2018.pdf) (Dhekne+ MobiSys'18) | UWB radio through container | Wave | Active | Liquid ID (permittivity) | Physics + classifier | 2 |
| 51 | [Food and Liquid Sensing in Practical Environments using RFIDs](https://www.usenix.org/conference/nsdi20/presentation/ha) (RF-EATS, Ha+ NSDI'20) | RFID on container (near field) | Response | Active | Content class | Deep | 2 |
| 52 | [Deep Neural Object Analysis by Interactive Auditory Exploration with a Humanoid Robot](https://arxiv.org/abs/1807.01035) (Eppe+ IROS'18) | Robot ears, shaken container | Source | Robot shaking | Material, weight | Deep | 2 / 3 |
| 53 | [Self-Supervised Moving Vehicle Tracking With Stereo Sound](https://openaccess.thecvf.com/content_ICCV_2019/html/Gan_Self-Supervised_Moving_Vehicle_Tracking_With_Stereo_Sound_ICCV_2019_paper.html) (Gan+ ICCV'19) | Stereo mics | Source | Passive | **Image-space** boxes | Deep, vision teacher → audio student | 5 |
| 54 | [Semantic Object Prediction and Spatial Sound Super-Resolution with Binaural Sounds](https://arxiv.org/abs/2003.04210) (Vasudevan+ ECCV'20) | 8 binaural mics | Source | Passive | **Image-space** semantic masks + depth | Deep, vision teacher → audio student | 5 |
| 55 | Seeing through Sounds: Predicting Visual Semantic Segmentation Results from Multichannel Audio Signals (Irie+ ICASSP'19) | Mic array | Source | Passive | **Image-space** person masks | Deep, camera-supervised | 5 |
| 56 | [Ambient Sound Provides Supervision for Visual Learning](https://arxiv.org/abs/1608.07017) (Owens+ ECCV'16) | Video + audio | — | Passive | Representation | Deep, cross-modal self-sup. | 5 |
| 57 | [Through Fog High-Resolution Imaging Using Millimeter Wave Radar](https://arxiv.org/abs/1912.09579) (HawkEye, Guan+ CVPR'20) | mmWave heatmaps | Wave | Active | **Depth image** (car shape) | Deep (cGAN), stereo-camera supervised + sim | 5 / 1 |
| 58 | [Image Reconstruction by Domain-Transform Manifold Learning](https://arxiv.org/abs/1704.08841) (AUTOMAP, Zhu+ Nature'18) | MRI k-space (any sensor domain) | — | Active | **Image** | Deep, direct sensor → image | 5 / 1 |
| 59 | [Deep D-bar: Real-Time Electrical Impedance Tomography Imaging with Deep Neural Networks](https://ar5iv.arxiv.org/html/1711.03180) (Hamilton & Hauptmann, IEEE TMI'18) | Boundary voltages (EIT) | Response | Active (current injection) | **Conductivity image** | Deep + physics (post-processing) | 5 / 1 |
| 60 | [OpenFWI: Large-scale Multi-structural Benchmark Datasets for Full Waveform Inversion](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27d3ef263c7cb8d542c4f9815a49b69b-Abstract.html) (Deng+ NeurIPS'22 D&B) | Seismic waveforms (sim) | Wave | Active | **Velocity map** | Deep (InversionNet etc.), benchmark | 5 / 1 |
| 61 | [Deep Learning Techniques for Inverse Problems in Imaging](https://arxiv.org/abs/2005.06001) (Ongie+ IEEE JSAIT'20) | Review | — | — | Image | Review: taxonomy by forward-model use | 5 |
| 62 | [Lamphone: Passive Sound Recovery from a Desk Lamp's Light Bulb Vibrations](https://www.usenix.org/conference/usenixsecurity22/presentation/nassi) (Nassi+ USENIX Sec'22) | Photodiode + telescope | Recovery | Passive | Audio | None | 4 |
| 63 | [Thermal Non-Line-of-Sight Imaging](https://visual.ee.ucla.edu/TNLOS_ICCP19.htm) (Maeda+ ICCP'19) | Long-wave IR camera | Wave (reflection) | Passive (object heat) | 2D shape, 3D location | Physics | 1 / 2 |
| 64 | [A Survey on Wi-Fi Sensing Generalizability](https://arxiv.org/abs/2503.08008) (IEEE COMST'25) | Wi-Fi (survey) | Wave | Active | — | Survey: domain shift across rooms, devices, people | 1 |
| 65 | [Characterization and Reduction of Artifacts in Limited Angle Tomography](https://www.math.uni-luebeck.de/veranstaltungen/vortraege/frikel.php) (Frikel & Quinto, Inverse Problems'13 ?) | X-ray CT (theory) | Wave | Active | — | Theory: missing angles → streaks | 1 |
| 66 | [Experimental Analysis of the Acoustic Field of Air-Coupled Ultrasound](https://opus4.kobv.de/opus4-bam/files/52386/Buehling_et_al_+2021_Experimental_analysis_of_the_acoustic_field.pdf) (Buehling+ BAM'21) | Air-coupled ultrasound | Wave | Active | — | Air/solid impedance mismatch | 1 |
| 67 | [HPS: X-ray security screening FAQ](https://hps.org/ate_faq/securityscreening/) (Health Physics Society) | X-ray | Wave | Active | — | Shielding / dose regulation (non-paper; replace with a proper cite) | 1 |
| 68 | [Active Acoustic Contact Sensing for Soft Pneumatic Actuators](https://ieeexplore.ieee.org/document/9196916) (Zöller, Wall, Brock ICRA'20; IJRR'23 ext. [arXiv](https://arxiv.org/abs/2208.10299)) | Embedded speaker + mic | Response | Active sweep | Contact location, force, material | Classical ML (SVM) | 3 |
| 69 | [See, Hear, and Feel: Smart Sensory Fusion for Robotic Manipulation](https://arxiv.org/abs/2212.03858) (Li+ CoRL'22) | Contact mic + camera + GelSight | Source | Manipulation contact | Policy | Deep (self-attention fusion) | 3 |
| 70 | [Hearing Touch: Audio-Visual Pretraining for Contact-Rich Manipulation](https://arxiv.org/abs/2405.08576) (Mejia+ ICRA'24) | Contact mic + camera | Source | Manipulation contact | Policy | Deep (AV pretraining) | 3 |
| 71 | [ManiWAV: Learning Robot Manipulation from In-the-Wild Audio-Visual Data](https://arxiv.org/abs/2406.19464) (Liu+ CoRL'24) | Contact mic in gripper finger | Source | Manipulation contact | Policy | Deep (diffusion policy) | 3 |

## Clusters (finer than sections; row numbers)

| Cluster | What it is | Rows | Gap it leaves for us |
|---|---|---|---|
| A. Wave imaging of hidden scenes | RF / mmWave / acoustic / optical NLOS; resolution set by wavelength and aperture | 4, 5, 6, 11, 34, 35, 37, 57 | Needs penetrating waves and an aperture; we use the enclosure's own low-frequency modes |
| B. Camera-supervised non-visual → image | Camera as teacher, other sensor as student | 2, 7, 8, 9, 38, 39, 53, 54, 55, 56, 57 | All wave or source regime; none in the response regime |
| C. Sound of impacts / interaction | Object emits its own signature | 12–15, 18, 19, 47, 52 | Needs contact or impact with the *object*; ours never touches it |
| D. Active response sensing (ubicomp) | Excite a fixed object, classify the perturbation | 16, 17, 46, 51 | Output stays a class |
| E. Inverse vibration / SHM / mass localization | Modal changes → damage or mass location | 20, 21, 22, 43, 44, 45 | Scalars or locations, mostly sim or model-based; identifiability theory (45) |
| F. Active excitation + laser/speckle readout (NDT) | Drive with sound or force, scan the surface optically, find the hidden thing | 40, 41, 42 | Closest physical precedent; imaging is a raw vibration map, not learned, and not an object image |
| G. Optical vibration sensing (recovery) | Recover the signal itself | 24–27, 33, 48, 49, 62 | Goal is the signal or an activity class, not the scene |
| H. Visual vibrometry / material from motion | Modes from video → material properties | 23, 30, 31, 32, 36 | Properties of the vibrating object, not its contents |
| I. Through-container content sensing | What's inside a sealed container | 3, 50, 51, 52 | Contents class (liquid, fill); no spatial layout |
| J. Learned inverse problems / tomography | Learn the sensor → image inverse | 58, 59, 60, 61 | Forward model is known (MRI, EIT, FWI); ours has none, so we learn from camera labels |
| K. Semantic audio → image | Plausible image, no geometry | 28, 29 | No physical grounding |
| L. Room acoustics response | Fixed room, person changes the RIR | 10 | Room-scale, location only |

## Narrative options for Related Work

**N1. Taxonomy-led ("by regime")**: wave → source → response, then the empty cell.
1. *Seeing with waves* (A, B wave rows): time-of-flight imaging works when there are penetrating waves and an aperture.
2. *Listening to objects* (C, H): impacts and visual vibrometry infer object properties, but need contact or line of sight to the object.
3. *Sensing by perturbation* (D, E, F, I, L): the object changes a fixed system's response. This is mature for classification and SHM localization; imaging exists only as raw NDT vibration maps.
4. Close with Table 1 (the regime × output grid): ours is the only response-regime entry with image output.
- Pros: clean and defensible, and it reuses the existing Regime column. Cons: reads like a survey, and the ML contribution shows up only at the end.

**N2. Sensor → measurement → learning funnel (ML-venue framing)**
1. *Optical vibration sensing* (G, 4): speckle and LDV now give multi-point kHz vibration cheaply. That is the enabling hardware.
2. *Vibration as a probe of what's attached* (E, F, H, D, I): decades of SHM, NDT and ubicomp show that the response encodes mass and contact, but outputs are classes, scalars or raw maps. Kichler+ (3) is the nearest: same sensor, class output.
3. *Learning to map non-visual signals to images* (B, J, K): camera-as-teacher works for RF, echoes and binaural sound, and learned inverse problems work when a forward model exists. We have neither a wave model nor a forward model, so camera supervision is the only inverse.
- Pros: ends on our contribution, which suits CVPR/NeurIPS. Cons: hidden-scene radar (A) gets squeezed in as a side note.

**N3. Problem-led ("seeing inside a closed box")**
1. *Penetrate it* (A, I-RF): mmWave, RF and acoustic NLOS image or identify contents, but need wavelength-scale aperture or special hardware, and Wave-Former/GeRaF give 3D but need radar.
2. *Shake or knock it* (C, I-shaking, 3): contents change the sound or vibration, giving material, fill level or count but not where things are.
3. *Treat the box as the sensor* (E, F, 45, 40, 41): mass on a plate shifts its modes (Kac: identifiability is not guaranteed). Acoustic-to-seismic LDV and vibro-acoustography already scan a driven surface to find buried things, without learning and without an object image.
4. *Learn the inverse* (B, J): with no tractable forward model, we borrow cross-modal supervision.
- Pros: the strongest story. Each paragraph ends with the limitation the next one fixes, and the box-as-sensor idea gets a lineage. Cons: the sections cut across the existing 5-section structure.
