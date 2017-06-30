import pandas as pd
import pytest

import organize


nan = float('nan')

def test_parse_sync_lines():
    lines = [
        'ignored line',
        '<Sync time="0"/>Start',
        '<Sync time="1.0"/>Some content',
        '<Sync time="2.0"/>End',  # Should be ignored
    ]
    expected = [
        (0, 'Start'),
        (1.0, 'Some content'),
    ]

    assert list(organize.parse_sync_lines(lines)) == expected


def test_parse_time_slots_lines():
    lines = [
        '    <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="7100"/>\n',
        '    <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="11520"/>\n',
        '    <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="12390"/>\n',
        '    <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="12990"/>\n',
    ]

    expected = [
        (0, False),  # Initialized to False
        (7.1, True),
        (11.52, False),
        (12.39, True),
        (12.99, False),
    ]

    assert list(organize.parse_time_slots_lines(lines)) == expected


def test_concat_by_time():
    catering = [
        (0, 'food'),
        (2, 'drinks'),
    ]
    speaker = [
        (1, 'Moshe'),
        (3, 'Yossi'),
    ]
    ac = [
        (1, False),
        (2, True),
    ]

    data = [
        [0, 'food', nan, nan],
        [1, nan, 'Moshe', False],
        [2, 'drinks', nan, True],
        [3, nan, 'Yossi', nan],
    ]
    columns = [
        'time',
        'catering',
        'speaker',
        'ac'
    ]
    expected = pd.DataFrame(data, columns=columns).set_index('time')

    result = organize.concat_by_time(catering=catering, speaker=speaker, ac=ac)
    assert result.equals(expected)


def test_concat_by_time_with_duplicate_indices():
    a = [
        (1, 'A'),
        (1, 'B'),
        (2, 'C'),
    ]
    b = [
        (1, 'X'),
        (3, 'Y'),
        (3, 'Z'),
    ]

    data = [
        [1, 'A', 'X'],
        [1, 'B', 'X'],
        [2, 'C', nan],
        [3, nan, 'Y'],
        [3, nan, 'Z'],
    ]
    expected = pd.DataFrame(data, columns=['time', 'a', 'b']).set_index('time')

    assert organize.concat_by_time(a=a, b=b).equals(expected)


def test_find_needed_files():
    filenames = [
        'asd(speak).trs',
        'qweS.eye.eaf',
        'zxcL.nod.eaf',
        'another_file.txt',
    ]
    expected = tuple(filenames[:3])
    assert organize.find_needed_files(filenames) == expected


def test_find_needed_files_raise():
    filenames = [
        'asdasd(speak).trs',
        'qweqweS.eye.eaf',
    ]
    with pytest.raises(FileNotFoundError):
        organize.find_needed_files(filenames)


def test_properly_annotated():
    df = pd.DataFrame([
        ['whatever', 'whatever'],
        [1, nan],
        [nan, 1],
    ])
    assert organize.properly_annotated(df)


def test_not_properly_annotated():
    df = pd.DataFrame([
        ['whatever', 'whatever'],
        [1, nan],
        [2, nan],
    ])
    assert not organize.properly_annotated(df)


def test_parse_interaction_number():
    examples = [
        ('data/rapport-oct-2006-all-transcriptions/20061013_121032-10-13-2006-SES5.R.N', 5),
        ('data/rapport-oct-2006-all-transcriptions/20061107_160243-11-07-2006-SES48.M.NN', 48),
        ('data/f2f-2007-all-transcriptions/20071126_093844-11-26-2007-Rapport.SES156.F.E.CC', 156),
    ]
    for arg, expected in examples:
        assert organize.parse_interaction_number(arg) == expected


def test_set_interaction_number():
    data = [
        [0, 'Start', False],
        [1, 'event', True],
    ]
    df = pd.DataFrame(data, columns=['time', 'events', 'nod']).set_index('time')

    df = organize.set_interaction_index(df, 123)

    assert len(df.index.levels) == 2
    for index_value in df.index.get_level_values(0):
        assert index_value == 123
