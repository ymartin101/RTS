# SOARS - Space Object and Astrodynamics Radar Simulator

---

SOARS is a ray-traced radar simulator for simulating the performance and output of a variety of radar systems. It is based on the design of the Flexible Extensible Radar Simulator (FERS) as well as the Ray Tracing Simulator (RTS), and could be used to generate useful results for real-world projects involving space-oriented systems.

#### AUTHORS

SOARS was namely developed by:

* Yaaseen Martin
* Marc Brooker (original FERS design)
* Kathryn Williams (original RTS design)

#### BUILDING SOARS

SOARS depends on a number of external libraries that you need to install before attempting to build the project. The libraries that you need to install are:

* Boost
* FFTW3
* HDF5 (libhdf5)
* PugiXML
* CMake

Additionally, SOARS requires the installation of a C++ compiler, NVIDIA CUDA, and Python. NVIDIA OptiX will also be required for the ray-tracing implementation, and the steps below will explain how to get this working.

The following steps will enable you to install SOARS:

* Download and extract the OptiX 5.1.1. software (linked here), then extract it to a folder "ROOT" (rename as desired)
* Download the SOARS repository and copy all its files into the ROOT/SDK/ directory; overwrite any conflicting files
* Change directory to ROOT/SDK/ and create a "build" folder, then change directory into it
* Inside ROOT/SDK/build, run "cmake3 .." (or just "cmake ..", depending on your installation)
* Run "make" to build the completed binary
* Place your input file (.soarsxml) into the "build" directory and then run it SOARS using "./soars INPUT.soarsxml".

Note that SOARS was only tested on a compute node running CentOS 7 with a NVIDIA V100 GPU; however, the program is written in standard C++ and should thus compile and run on many architectures and operating systems.

#### SOARS USAGE

SOARS takes an XML description of a scene, with one or more transmitters, one or more receivers, and zero or more targets. Using the parameters (such as sample rate, carrier frequency, etc.) defined in the XML script, SOARS generates the waveform that the receivers will observe after passing through an ADC.

SOARS works with an arbitrary number of receivers and transmitters (monostatic and multistatic) for pulsed radar systems using various pulse shapes. The simulator currently simulates the amplitude, phase, Doppler and noise effects of a radar system. Simulation of ray-traced effects (such as multiscatter, refraction, etc) is also supported through NVIDIA OptiX.

In terms of inputs, SOARS requires a single XML file (with the extension ".soarsxml") that describes the entire simulation. It then outputs:

* One or more files representing the raw radar return at the terminals of each receiver in the simulation (in HDF5 format)
* Optional .soarsxml file(s) containing additional information about the simulation

Note that:

* The values in the files represent voltages across a 1 ohm resistor. Thus, to calculate power, one takes the square of the values in the data.
* All input and output signals are in baseband i.e., the carrier component is implicit. The Nyquist sampling rate (also specified in the .soarsxml input file) is thus lowered and you do not need to sample at twice the carrier frequency.
* The return signals are normalised and each has a scaling factor. To obtain the actual values one must thus multiply the data by this scaling value.

#### THANKS

The authors of PugiXML, FFTW3, CMake, HDF5, FERS, RTS, SGP4, OptiX, CUDA, and Boost.

#### COPYRIGHT NOTICE

SOARS is covered by the following copyright notice. Should you wish to acquire a copy of FERS not covered by these terms, please contact the Department of Electrical Engineering at the University of Cape Town.

Please note that this copyright only covers the source code, program binaries and build system of SOARS. Any input files you create and results created by the simulator are not covered by this notice and remain copyright of their original author. Also, this copyright notice does not cover the source code of any external dependencies used as part of this repository (such as SGP4).

SOARS - Space Object and Astrodynamics Radar Simulator

Copyright (C) 2022 Yaaseen Martin

This program is free software; you can redistribute it and/or modify
it under the terms of version 3 of the GNU General Public License as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA
