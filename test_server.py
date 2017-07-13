import flask
import pytest
import requests

import server


@pytest.fixture
def app():
    app = server.app
    app.debug = True
    return app


@pytest.mark.usefixtures('live_server')
def test_server_not_completely_broken():
    j = {
        'listeners':
        {
            1: [0, 1],  # id: [speaker_silent, speaker_gaze]
        },
    }
    url = flask.url_for('backchannel_handler', _external=True)
    resp = requests.post(url, json=j)
    assert resp.status_code == 200


@pytest.mark.usefixtures('live_server')
def test_server_with_dekok():
    j = {
        'type': 'dekok',
        'listeners':
        {
            1: [0, 1],
        },
    }
    url = flask.url_for('backchannel_handler', _external=True)
    resp = requests.post(url, json=j)
    assert resp.status_code == 200
