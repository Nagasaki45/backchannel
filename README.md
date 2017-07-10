# Automated backchannel behaviour

This work is part of my UnsocialVR project (TODO reference).
Here, I'm using the ICT Rapport Datasets to learn backchannel behaviours, and build a model to generate similar active listening responses in automated agents.

## Getting started

- Install dependencies with `conda env create -f environment.yml`.
- Activate environment `source activate backchannel`.
- Download the datasets manually (see below) to, let's say, `mydata`.
- Organize the datasets with `python organize.py mydata`.
- Run `train.ipynb` to train, investigate, and persist the model.

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

The results are written to `data.hdf`, with interaction number as keys. The columns of each table are:

- `time` (timedelta)
- `speaker_behaviour` (string).
- `speaker_eye` (bool)
- `listener_nod` (bool)

## `train.ipynb`

Train the ML model and see how it performs on test data.
The model is pickled into `model.pickle`.

## Running tests

```bash
py.test
```
