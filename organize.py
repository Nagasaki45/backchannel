"""
Organize data from the ICT Rapport Datasets to train automatic side
participant agent.
"""

import argparse
import itertools
import logging
import os
import re
import sys

import pandas as pd

SYNC_PATTERN = re.compile(r'<Sync time="(-?\d+.?\d*)"/>(.+)$')
TIME_SLOT_PATTERN = re.compile(r'<TIME_SLOT TIME_SLOT_ID="ts\d+" TIME_VALUE="(\d+)"/>')
INTERACTION_NUMBER_PATTERN = re.compile('SES(\d+).')
IGNORED_SYNC_LINES = ['end']
SPEAKER_BEHAVIOUR_END = '(speak).trs'
SPEAKER_EYE_END = 'S.eye.eaf'
LISTENER_NOD_END = 'L.nod.eaf'
REQUIRED_ENDINGS = [SPEAKER_BEHAVIOUR_END, SPEAKER_EYE_END, LISTENER_NOD_END]
OUTFILE = 'data.csv'


def parse_sync_lines(lines):
    """
    Parse sync lines from speaker behaviour XML annotation files.
    """
    for line in lines:
        match = SYNC_PATTERN.search(line)
        if match:
            time, content = match.groups()
            if content.lower() not in IGNORED_SYNC_LINES:
                yield float(time), content


def parse_time_slots_lines(lines):
    """
    Parse time slots lines from speaker eye gaze and listener head nods XML
    annotation files.
    NOTE: Assuming that time slot lines start and stop the behaviour, ignoring
    the actual ANNOTATION tag that link between them.
    """
    yield (0, False)

    state = True
    for line in lines:
        match = TIME_SLOT_PATTERN.search(line)
        if match:
            millis = int(match.groups()[0])
            yield millis / 1000, state
            state = not state


def concat_by_time(**kwargs):
    """
    Builds a pd.DataFrame with data from lists of (time, value), with the
    name of the list correspond to the name of the column.
    """
    name_data_iter = iter(kwargs.items())
    name, data = next(name_data_iter)
    df = _partial_df(name, data)
    for name, data in name_data_iter:
        df = df.merge(_partial_df(name, data), how='outer',
                      left_index=True, right_index=True)
    return df


def _partial_df(name, data):
    return (pd
            .DataFrame(data, columns=['time', name])
            .set_index('time'))


def organize_interaction_data(speaker_behaviour, speaker_eye, listener_nod):
    """
    Given the content of the 3 files (list of lines) create a pd.DataFrame
    with the data.
    """
    speaker_behaviour = list(parse_sync_lines(speaker_behaviour))
    speaker_eye = list(parse_time_slots_lines(speaker_eye))
    listener_nod = list(parse_time_slots_lines(listener_nod))
    return concat_by_time(
        speaker_behaviour=speaker_behaviour,
        speaker_eye=speaker_eye,
        listener_nod=listener_nod
    )


def organize_data(root_dir):
    """
    Traverse the datasets directory and build one big dataframe for all
    properly annotated interactions.
    """

    dfs = []

    for (dirpath, _dirnames, filenames) in os.walk(root_dir):
        try:
            found_files = find_needed_files(filenames)
        except FileNotFoundError:
            logging.debug(f'Needed files missing from "{dirpath}". Ignoring')
            continue

        speaker_behaviour_f, speaker_eye_f, listener_nod_f = found_files
        speaker_behaviour = _read_lines(dirpath, speaker_behaviour_f)
        speaker_eye = _read_lines(dirpath, speaker_eye_f)
        listener_nod = _read_lines(dirpath, listener_nod_f)

        df = organize_interaction_data(
            speaker_behaviour,
            speaker_eye,
            listener_nod
        )

        if not properly_annotated(df):
            msg = f'Interaction "{dirpath}" is not properly annotated. Ignoring'
            logging.debug(msg)
            continue

        df = set_interaction_index(df, parse_interaction_number(dirpath))

        dfs.append(df)

    return pd.concat(dfs)


def _read_lines(dirpath, filename):
    with open(os.path.join(dirpath, filename)) as f:
        return f.readlines()


def find_needed_files(filenames):
    """
    Return the filenames of the needed files or raises FileNotFoundError.
    """
    results = []
    for ending in REQUIRED_ENDINGS:
        for filename in filenames:
            if filename.endswith(ending):
                results.append(filename)
                break
        else:
            raise FileNotFoundError(f'No file match {ending}')
    return tuple(results)


def properly_annotated(df):
    """
    Return True if the pd.DataFrame is properly annotated. Otherwise False.
    """
    for column in df.columns:
        if df[column].drop(0).isnull().all():
            return False
    return True


def parse_interaction_number(dirpath):
    match = INTERACTION_NUMBER_PATTERN.search(dirpath)
    return int(match.groups()[0])


def set_interaction_index(df, interaction_number):
    """
    Add a top level index to the dataframe with the interaction number.
    """
    df['interaction'] = interaction_number
    return (df
            .set_index('interaction', append=True)
            .reorder_levels(['interaction', 'time']))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets_dir', help='root directory of the datasets')
    parser.add_argument('--verbose', '-v',
                        action='store_true', help='verbose output')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    df = organize_data(args.datasets_dir)
    df.to_csv(OUTFILE)


if __name__ == '__main__':
    main()
