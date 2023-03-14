# RTS - Ray Tracing Simulator

---

The RTS is a ray-tracing algorithm built on NVIDIA OptiX for use in signal-level radar simulation. This is able to perform ray tracing of a transmitted radar signal, reflect and refract the signal based on targets in an environment, and then collect received rays at simulated receivers for post-processing of the recorded ray quantities (such as power, phase, length, and more).

Please see SOARS for how this was implemented for a typical existing radar simulator. The base code may need to be adapted to fit certain simulation schemes, but the idea is for the RTS to be modular in its design. Please also note that the RTS cannot be compiled or run on its own and must be retroactively refitted into an existing simulator.

#### AUTHORS

The RTS was namely developed by:

* Yaaseen Martin
* Kathryn Williams (original RTS design)

#### THANKS

The authors of PugiXML, FFTW3, CMake, HDF5, FERS, RTS, SGP4, OptiX, CUDA, and Boost.

#### COPYRIGHT NOTICE

SOARS is covered by the following copyright notice. Should you wish to acquire a copy of FERS not covered by these terms, please contact the Department of Electrical Engineering at the University of Cape Town.

Please note that this copyright only covers the source code, program binaries and build system of SOARS. Any input files you create and results created by the simulator are not covered by this notice and remain copyright of their original author. Also, this copyright notice does not cover the source code of any external dependencies used as part of this repository.

Copyright (C) 2023 Yaaseen Martin

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
