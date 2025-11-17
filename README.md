# High‑Quality RAW Image Processing & HDR Pipeline

A complete end‑to‑end digital imaging pipeline implemented **from scratch**, covering demosaicing, tone mapping, white‑balance, HDR fusion, and iCAM06. This project reproduces a miniature **digital camera RAW processor** and **HDR engine** using only low‑level algorithms taught in the lecture, without relying on built‑in camera pipelines.

The entire pipeline processes **linear sensor data** and reconstructs visually appealing images while preserving physical correctness.

---

## ✨ Highlights

* Full-fledged RAW → RGB pipeline coded manually
* Bayer pattern decoding & bilinear demosaicing (no library demosaic)
* Physically meaningful gamma/log tone mappings
* Gray-world white balance (lecture-based)
* Exposure-linear HDR fusion from RAW images
* iCAM06 tone mapping (base/detail + bilateral decomposition)
* High‑quality CR3 → JPG converter (`process_raw`, quality=99)
* Optional pipeline for HDR from JPG via camera response curve inversion

This README is crafted to reflect research-grade understanding and engineering quality.

---

## 1. Bayer Pattern Investigation

Identified the Bayer pattern from raw sensor data to ensure correct CFA interpretation.

---

## 2. Bilinear Demosaicing 

Implemented a basic bilinear demosaicer using CFA masks and a 3×3 averaging kernel to reconstruct RGB.

---

## 3. Global Tone Enhancement (

Applied percentile-based normalization and two simple tone curves (gamma and log).

---

## 4. White Balance – Gray World 

Used gray-world gains to equalize channel averages.

---

## 5. Sensor Linearity Verification

Six images with halved exposures (1/10 → 1/320) were analyzed:

* Compute mean RAW intensity of each image
* Plot intensity vs exposure time
* Result is a perfect straight line → **proves sensor response is linear**

This is essential for RAW‑domain HDR merging.

---

## 6. HDR Merging 

Implements textbook HDR from RAW:

* Scale each RAW frame by exposure ratio (each is ×2 brighter than the next)
* Replace saturated pixels of longer exposures with scaled values from shorter ones
* Combine all exposures into a **single linear HDR frame**
* Perform demosaicing + white balance afterward

This creates physically correct HDR without camera curves.

---

## 7. iCAM06 Tone Mapping 

Full implementation of iCAM06:

1. Convert RGB → intensity + chroma
2. Bilateral filtering of log-intensity to obtain base + detail layers
3. Compress base layer using:

   ```text
   compression = log(output_range) / (max(log_base) - min(log_base))
   ```
4. Recombine chroma and detail layers
5. Produce visually pleasing LDR output

iCAM06 is a perceptually motivated tone-mapping operator used in HDR rendering.

---

## 8. Final Deliverable 

A polished RAW→JPG pipeline built entirely from custom components (no built-in postprocessing):

* demosaic_image 

* improve_luminosity_linear 

* gray_world white balance 

* optional warm color shift

* saturation enhancement (HSV domain)

* local contrast enhancement (unsharp masking)

* percentile-based normalization

* saved as high-quality JPG (quality=99)

---

## Summary

This project reimplements the essential components of a digital camera pipeline and HDR imaging system **directly from first principles**:

* RAW color filter array decoding
* Linear color reconstruction
* Dynamic-range manipulation
* Color constancy algorithms
* Multi-exposure fusion
* Perceptually motivated tone mapping

The pipeline produces physically meaningful intermediate results and high‑quality final images, demonstrating both engineering precision and conceptual understanding.
