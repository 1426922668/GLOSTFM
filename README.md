# GLOSTFM
This repository contains the implementation of the Global Spatiotemporal Fusion Model (GLOSTFM), a novel approach designed to generate global Land Surface Temperature (LST) products by integrating data from multiple remote sensing sources. GLOSTFM leverages the strengths of both thermal infrared and microwave sensors, effectively addressing challenges related to spatiotemporal continuity, computational efficiency, and data uncertainty on a global scale.

## Overview
GLOSTFM is built on image pyramid principles and utilizes data from China's Fengyun-3D (FY-3D) satellite, which is equipped with both thermal infrared (MERSI, 1 km) and microwave (MWRI, 25 km) sensors. By employing spatiotemporal fusion techniques, GLOSTFM enhances LST data continuity and accuracy, making it a powerful tool for global climate change research and monitoring applications, such as urban heat island analysis.

## Important Notice
GLOSTFM is a complex algorithm that requires proper adaptation based on the characteristics of the input data format and engineering constraints. The provided implementation serves as a reference example, but actual deployment should be tailored to specific application requirements and optimized according to the use case. Users should select the most appropriate algorithm configuration for their scenario.
For further theoretical background and methodology, please refer to the original research article:
DOI: https://doi.org/10.1016/j.rse.2025.114640
Hope this repository helps your research and development! 🚀
