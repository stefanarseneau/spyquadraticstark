## The Quadratic Stark Effect Is A Significant Source Of Bias

**10 Jan 2025:** *[Update: It appears that this was happening because I was running the validation checks on all the datapoints, not all the good datapoints. Apparently there's some randomness in failure modes, so spectra which couldn't be fit were flagged as non-compliant.]* Refactored to run on SCC. The following points are failing the validation check.

* HS1334p0701_a_2001_06_18T01_05_40_all.dat.gz

**09 Jan 2025:** Rewrote the fitting script to identify the point where the old LTE RVs don't match the new LTE RVs.

**08 Jan 2025:** I modified `stark/utils.py.read_clean_lte` to temporarily use the LTE RVs generated with the old code while the new code runs. The only change between the old code and the new one was refactoring, so in principle it should be the same thing, but I need to remember to check that.
