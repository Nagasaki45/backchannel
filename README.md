# Automated backchannel behaviour

This work is part of my UnSocialVR project (TODO reference).
Here, I'm using the ICT Rapport Datasets to learn backchannel behaviours, and build a model to generate similar active listening responses in automated agents.

## Datasets

Before starting go and grab the data from [the official ICT Rapport Datasets webpage](http://rapport.ict.usc.edu/).
You will need the two files:

- `Rapport October 2006 -> Session = ALL -> Transcription / Annotation .zip file`.
- `Face To Face 2007 -> Session = ALL -> Transcription / Annotation .zip file`.

Extract the archives to somewhere you know.

## `organize.py`

Organize data from the ICT Rapport Datasets to train automatic side
participant agent.

The script uses:

- Speaker behaviour annotations from `*(speak).trs` files.
- Speaker eye gaze (is looking at listener or not) annotations from `*S.eye.eaf` files.
- Listener head nods (is nodding or not) annotations from `*L.nod.eaf` files.

The result is a `data.csv` file in the current directory with skeleton data
to train the agent. The columns are:

- interaction_num
- time (in seconds)
- One-hot encoding of the speaker behaviour labels.
- speaker_eye
- listener_nod

Most of the cells will be N/A, because the time slots originate from 3
different files. However, they can be forward-filled easily.
