
# PyPhasePick

This repository contains scripts and a collection of functions to process and pick surface wave phase velocities extracted from seismic ambient noise.

The scripts are numbered 0 to 6 and represent a workflow for picking surface wave phase velocities. Each of the scripts does the following:
 * 0 - Takes the output cross correlations from an [MSNOISE](http://www.msnoise.org/) style directory and stacks them to produced Empirical Green's Functions (EGF) and (optionaly) stacked cross correlations.
 * 1 - Takes the previously computed EGF and converts it into a FTAN matrix which is saved as a set of xarray DataArrays.
 * 2 - Takes either the FTAN matricies or the EGFs and produced a regional reference curve that the phase velociteis are picked from.
 * 3 - Picks phase velocities in the velocity time domain using the regional reference curve.
 * 4 - Picks phase velociteis in the velocity frequency domain using seperately specified, higher frequency, reference curves that the main regional reference is not able to pick.
 * 5 - Loops through all of the high frequency picks and plots them giving an option to discard picks if they have not picked the correct one.
 * 6 - Optional manual picking script

The functions are in the pyphasepick directory and are sorted into three different modules
 * frequencytimeanalisys - Contains functions to perform Frequency Time Analysis.
 * picking - Contains functions to pick phase dispersion curves and produce the regional reference.
 * stackingandegf - Contains functions to produced stacked cross correlations and EGFs from the [MSNOISE](http://www.msnoise.org/) output.