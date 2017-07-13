import flask
import pytest
import requests

import server


@pytest.fixture
def app():
    return server.app


@pytest.mark.usefixtures('live_server')
def test_server_not_completely_broken():
    id_ = 1
    speaker_silent = 0
    speaker_gaze = 1
    url = flask.url_for('backchannel_handler', _external=True)
    resp = requests.post(url, json={id_: [speaker_silent, speaker_gaze]})
