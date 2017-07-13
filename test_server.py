import json

import flask
import pytest

import server


@pytest.fixture
def app():
    return server.app


def test_server_not_completely_broken(client):
    j = {
        'listeners':
        {
            1: [0, 1],  # id: [speaker_silent, speaker_gaze]
        },
    }
    url = flask.url_for('backchannel_handler')
    resp = client.post(url, data=json.dumps(j), content_type='application/json')
    assert resp.status_code == 200


def test_server_with_dekok(client):
    j = {
        'type': 'dekok',
        'listeners':
        {
            1: [0, 1],
        },
    }
    url = flask.url_for('backchannel_handler')
    resp = client.post(url, data=json.dumps(j), content_type='application/json')
    assert resp.status_code == 200
