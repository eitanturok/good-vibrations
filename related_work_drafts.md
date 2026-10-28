# Related Work: Draft A, three versions

Citation keys refer to rows in `lit_review.md`. Older drafts are below.

---

## A1. "What has to come back" (one question carries the section)

**Echo-based imaging.** Most ways of seeing hidden objects send a wave in and wait for it to return: X-rays [CT], radio and millimeter waves [RF-Pose, mmNorm, Wave-Former, GeRaF], and sound [Lindell+19, BatVision]. Whatever comes back is what you see, so the wave has to get through the occluder and back out to the receiver. Each wave pays for this differently. X-rays are ionizing. Radio and millimeter waves reflect off a surface like a mirror, miss the parts that face away from the antenna, and are blocked by metal. Airborne sound barely enters a solid at all. We drive the enclosure with sound too, but we never need that sound to come back. What returns to our sensor is laser light from the outside of the box.

**Contact vibration sensing.** If the probe never has to return, the enclosure itself has to carry the information, as a change in how it vibrates. This idea is old and widely used. Structural health monitoring locates damage from shifts in a structure's vibration modes [Doebling+96, Avci+21], impact sounds reveal shape and material [Zhang+17, ObjectFolder], and contact speakers turn everyday objects into touch sensors [Touch & Activate]. The Boombox [Chen+21] even reconstructs an image of an object dropped into a container. All of these need sensors attached to the surface, and most need the object to strike it.

**Remote vibration sensing.** Optical sensing removes that contact. Cameras and speckle recover sound from distant surfaces [Davis+14, Zalevsky+09, Sheinin+22], and a laser scanning sound-driven ground finds buried landmines [Sabatier & Xiang 01]. Kichler et al. [Kichler+25] use our sensor to classify the fill level of opaque containers. So far, remote vibration gives back a signal, a class, or a location, never an image.

**Learning the inverse.** Nothing models how an object inside a box changes the box's vibration well enough to invert. We therefore learn the mapping from paired camera images, as has been done for radio [RF-Pose, Person-in-WiFi], radar [HawkEye], and microphone arrays [Gan+19, Vasudevan+20].

---

## A2. Trade-offs (each paragraph fixes the last one's limitation)

**Echo-based imaging.** Seeing through a wall usually means sending a wave through it and timing the echo. This works well when a suitable wave exists. Radio senses people behind walls [Adib & Katabi 13, RF-Pose], millimeter-wave radar reconstructs objects inside cardboard boxes [mmNorm, Wave-Former, GeRaF], and sound maps hidden rooms [Lindell+19, BatVision]. Because the echo must return to the receiver, each wave brings its own limits. X-rays need radiation shielding. Radio and millimeter waves see only surfaces that face the antenna and cannot enter metal. Sound reflects almost entirely at the air-to-solid boundary. Our enclosure is a metal box, which none of these waves can get through.

**Contact vibration sensing.** A different line of work asks the structure itself. An object changes how whatever holds it vibrates, which is how engineers find damage in bridges [Doebling+96, Avci+21] and how touch interfaces sense a grasp [Touch & Activate, Knocker]. When the object strikes the surface, the sound reveals even more: material, shape [Zhang+17, ObjectFolder], and, in the Boombox [Chen+21], an image of where it landed. These methods need sensors glued to the surface, and usually an impact.

**Remote vibration sensing.** Cameras and lasers can measure the same vibration from a distance. They recover speech from a bag of chips [Davis+14] or a light bulb [Lamphone], find landmines under sound-driven soil [Sabatier & Xiang 01], and, closest to us, classify how full a sealed container is [Kichler+25]. Remote sensing removes the contact, but so far the outputs are signals and labels.

**Learning images from non-visual signals.** We want an image, and no physical model maps box vibration to box contents. We take the route that worked for radio and sound [RF-Pose, HawkEye, Vasudevan+20]: a camera supervises training and is removed at test time.

---

## A3. Short version (about half a page)

Most through-wall imaging is echo-based. Radio, millimeter waves, X-rays, and sound are sent toward a hidden object, and the returning wave forms the image [RF-Pose, Wave-Former, GeRaF, Lindell+19]. The wave must therefore pass through the occluder twice. That rules out metal enclosures for radio, forces shielding for X-rays, and leaves gaps wherever a surface reflects the signal away from the receiver. We instead read the hidden object from the vibration of the enclosure itself. The sound we play never has to return; only laser light from the outer surface does.

Using vibration to sense what a structure holds is well established: in damage detection [Doebling+96, Avci+21], in touch sensing on everyday objects [Touch & Activate], and in recognizing objects from impact sounds [Zhang+17, ObjectFolder]. The Boombox [Chen+21] goes as far as an image, but it requires contact microphones and an object dropped into the container. Optical sensors measure vibration without contact [Davis+14, Sheinin+22, Sabatier & Xiang 01]. With the same speckle sensor as ours, Kichler et al. [Kichler+25] classify the contents of sealed containers. We combine remote vibration sensing with camera supervision, as earlier work did for radio and sound [RF-Pose, Vasudevan+20], to recover an image of the object inside.

---

# Older drafts

## Draft A v2 (detailed, superseded by A1–A3)

---

**Echo-based imaging.** The standard way to see a hidden object is to send a wave toward it and measure what returns. Every such method has the same requirement: the wave must get through the occluder, reach the object, and come back to a receiver. The choice of wave therefore decides what can be seen.

X-rays pass through almost anything, but they are ionizing and so need shielding, trained operators, and regulatory approval [HPS]. Computed tomography also needs views from all around the object, and missing angles leave streak artifacts [Frikel & Quinto 13]. In practice the object must be carried into a scanner and fit inside it. Radio waves pass through walls and have been used to recover human pose and silhouettes [Adib & Katabi 13, RF-Pose, Person-in-WiFi, DensePose-WiFi]. At Wi-Fi frequencies, however, wavelengths are several centimeters, systems depend on installed access points, and models degrade when moved to a new room, device, or person [Wi-Fi generalization survey 25]. Millimeter waves give finer resolution and reach inside boxes [mmNorm, Wave-Former, GeRaF], but at these wavelengths most surfaces act like mirrors. A surface that does not face the antenna reflects the signal away, so parts of the object are never observed. Current systems handle this by sweeping the radar with a robot arm to build a large aperture [mmNorm] or by completing the missing geometry with learned shape priors [Wave-Former]. Neither radio nor millimeter waves pass through metal. Sound has the opposite problem. More than 99.9% of airborne sound reflects at an air–solid boundary, so medical and industrial ultrasound need gel or water coupling [air-coupled NDT]. In-air absorption also grows with the square of frequency, which trades resolution for range. Audible echoes travel well in air, but they recover room-scale geometry with large speaker and microphone arrays and scans that take minutes [Lindell+19, BatVision, VisualEchoes]. Thermal cameras need the hidden object to be warmer than its surroundings, and they see around corners only by reflection, not through walls [Maeda+19].

Our method also relies on a returning signal, but of a different kind. We drive the enclosure with sound and never need that sound to come back. We read its effect from laser light scattered off the enclosure's outer surface, which is always in view. The readout does not penetrate anything, needs no aperture around the object, and does not depend on which way the object's surfaces face. The enclosure is part of the scene, not part of the instrument.

**Contact vibration sensing.** Vibration carries rich information about objects. Impact sounds reveal shape and material [Zhang+17, ObjectFolder, RealImpact], contact microphones let robots perceive what they hold [SonicSense, Vibro-Sense], and driving a surface with a contact speaker turns everyday objects into touch sensors [Touch & Activate, Knocker]. Structural health monitoring applies the same idea to buildings and bridges, locating damage or added mass from shifts in a structure's vibration modes [Doebling+96, Avci+21, JSV'13]. The Boombox [Chen+21] is the closest in output: it reconstructs an image of an object dropped into a container from contact microphones. These methods require sensors attached to the surface, and most require the object to make the sound by being struck or dropped. Their outputs are mostly classes, scalars, or single locations.

**Remote vibration sensing.** Optical sensors remove the contact. Video [Davis+14], defocused speckle [Zalevsky+09], and rolling-shutter or event cameras [Sheinin+22, Event2Audio, Bagon+26] recover sound from distant surfaces, and laser vibrometers recognize activities across a room [Vibrosight, VibroSense, Lamphone]. Remote vibration also reveals what lies behind a surface. Video recovers material stiffness and interior properties [Davis+15, Feng+22]. Scanning sound-driven ground with a laser vibrometer finds buried landmines [Sabatier & Xiang 01]. Kichler et al. [Kichler+25] use the same speckle sensor as ours to classify fill level inside opaque containers. Most of this work recovers the signal itself, or a class or a scalar. None produces an image of an object hidden behind the surface.

**Learning images from non-visual signals.** No tractable forward model links an enclosure's vibration to its contents, so we learn the inverse from paired camera images. Using a camera as the teacher during training has worked for radio [RF-Pose, Person-in-WiFi], radar [HawkEye], echoes [BatVision], and microphone arrays [Irie+19, Gan+19, Vasudevan+20]. Learned reconstruction in MRI, impedance tomography, and seismic imaging [AUTOMAP, Deep D-bar, OpenFWI] can fall back on a known physical model. We cannot, so the camera is our only supervision.

---

## Draft B: Seeing inside a closed box (problem-led)

**Looking through the walls.** One way to see inside a box is to use a wave that passes through it. Millimeter-wave radar reconstructs hidden objects in 3D [mmNorm, Wave-Former, GeRaF], radio senses people behind walls [Adib & Katabi 13, RF-Pose], and RF tags identify the contents of containers [LiquID, RF-EATS]. These systems work when the walls are transparent to the wave. They need dedicated antenna arrays, and their resolution is tied to wavelength and aperture.

**Shaking and knocking.** A second way is to make the contents produce a signal. Robots shake containers to estimate what and how much is inside [Eppe+18], impacts reveal material and shape [Zhang+17, ObjectFolder], and the Boombox [Chen+21] turns the sound of an object hitting a container into an image of where it landed. These methods tell us *what* is there, but they need the contents to move or be struck.

**The box as the sensor.** A third way leaves the contents alone and listens to the box. Any mass resting on a structure changes how that structure vibrates, a fact used to localize added mass on plates [mass-on-plates, JSV'13] and damage in buildings [Doebling+96, Avci+21]. Whether these changes determine shape is a classic question: one cannot always hear the shape of a drum [Kac 66, Gordon+92]. In practice, scanning a driven surface with a laser already finds buried landmines [Sabatier & Xiang 01], and ultrasound-driven vibration images hard inclusions in tissue [Fatemi & Greenleaf 98]. Kichler et al. [Kichler+25] bring this idea to containers with speckle vibrometry and classify fill level. None of these methods produces an image of the object inside.

**Learning the inverse.** We close this gap with learning. Because the relationship between contents and vibration is too complex to model, we supervise with a camera, as prior work has done for radio [RF-Pose, Person-in-WiFi], echoes [BatVision], and binaural sound [Vasudevan+20, Gan+19].

---

## Draft C: Learning to see from new signals (learning-led)

**Images from non-visual signals.** A growing body of work trains networks to produce images from signals a camera cannot see. A camera serves as the teacher during training and is removed at test time. This has produced human pose and masks from radio [RF-Pose, Person-in-WiFi, DensePose-WiFi], depth from echoes [BatVision, VisualEchoes], car shapes from radar [HawkEye], and object masks and positions from microphone arrays [Irie+19, Vasudevan+20, Gan+19]. When the physics is known, learning instead sharpens classical reconstruction, as in MRI, impedance tomography, and seismic inversion [AUTOMAP, Deep D-bar, OpenFWI]. Almost all of these signals are echoes: waves that travel to the object and back. Others are emissions, sounds the scene makes on its own. We study a third kind: the response of a structure to the object it holds.

**Vibration as a probe.** Objects change how the things they touch vibrate. This idea underlies structural health monitoring [Doebling+96, Avci+21], mass localization on plates [JSV'13], and touch sensing on everyday objects [Touch & Activate, VibSense]. Learning has made these signals more useful, but the outputs remain classes, scalars, or single locations. The Boombox [Chen+21] produces an image, but only from the sound of an object being dropped and only with contact microphones.

**Measuring vibration from afar.** Optical sensors make this signal practical. Video [Davis+14, Davis+15], speckle [Zalevsky+09, Sheinin+22], and laser vibrometry [Sabatier & Xiang 01, Vibrosight] measure surface vibration without contact, and now record many points at kilohertz rates. Most of this work recovers the signal itself, such as speech or music. Kichler et al. [Kichler+25] use it to classify what is inside a sealed container. We combine remote, multi-point vibration with camera supervision to produce what none of these methods do: an image of the hidden object.
