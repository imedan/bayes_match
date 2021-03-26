# Bayesian Cross-Match to Gaia DR2 Sources

This repository provides the code used to calculate the Bayesian probability of stars in some photometric catalog being a match to a subset of Gaia sources. Additionally, an example use of the code is provided in the form of a Jupyter Notebook. This code was written to carry out the analysis outlined in [Medan, Lepine and Hartman (2021)](https://arxiv.org/abs/2102.10210), not as a stand alone Python package. Because of this, if you would like to adapt or use this code, it is strongly encouraged that you read that paper to understand the assumptions made in this method. Additionally, using this code requires pre-querying an external catalog (as outlined in [Medan, Lepine and Hartman (2021)](https://arxiv.org/abs/2102.10210)) and formatting the data as stated in the example Jupyter Notebook.

If you use any of this code to match data used in a subsequent study, please cite [Medan, Lepine and Hartman (2021)](https://arxiv.org/abs/2102.10210).

# "Chunked" Match (In Development)

I am currently working on a "chunked" version of the cross-matching code. This may be good for large Gaia datasets (and thus external catalogs). Idea here is to divide the input Gaia catalog into chunks, and perform the steps in the method on each chunk. This, in combination with a few code changes/parallelizations, should 1) speed up the match and 2) allow someone to stop the most bulky parts of the method midway, and restart from whatever chunk you stopped at.

Also in this new version of the code, I automatically pre-query the external catalog using XMatch in astroquery, so this doesn't have to be done ahead of time (and you wont have to download a very large file pre-queried to such a large radius like before).

# Installation

To install this code onto your machine simply run:

	pip install git+https://github.com/imedan/bayes_match.git@chunked_match
