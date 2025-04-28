# vpi_apriltag_detection
Demo code to use NVidia VPI API to detect april tag. 

The code follows the VPI 3.2 samples from [Vision Programming Interface](https://docs.nvidia.com/vpi/samples.html). 


# Platform
The testing platform is Jetson Orin NX 16GB on Seeed Studio J401 carrier board. 

Please be noticed, Jetson Orin Nano has no PVA built-in, you will only be using the CPU backend not PVA (progarmmable vision accelerator). 

You can check "Technical Specifications" comparison table on [NVidia Jetson modules](https://developer.nvidia.com/embedded/jetson-modules) to see whether your board has PVA built-in or not (search for "Vision Accelerator"). 


# Pre-requesite
If you flash your Jetson system using Jetpack 6.1 or later, most likely you don't need to do anything else special.

Otherwise, please find the [Installation](https://docs.nvidia.com/vpi/installation.html) instructions.
