import logging

import pytest

from app.core.config import get_settings

ASK = "/api/v1/ask"


async def test_oversized_json_body_is_rejected_before_it_is_parsed(api_client):
    size = get_settings().max_request_body_kb * 1024
    body = b'{"question": "' + b"a" * size + b'"}'

    resp = await api_client.post(ASK, content=body, headers={"Content-Type": "application/json"})

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


async def test_malformed_json_is_a_422_that_does_not_echo_the_input(api_client, alice):
    resp = await api_client.post(
        ASK,
        content=b'{"question": "secret-text',
        headers={**alice, "Content-Type": "application/json"},
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"
    assert "secret-text" not in resp.text


async def test_wrong_method_uses_the_standard_error_shape(api_client, alice):
    resp = await api_client.patch("/api/v1/documents", headers=alice)

    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "method_not_allowed"
    assert "POST" in resp.headers["allow"]


@pytest.mark.parametrize("path", ["/api/v1/health", "/does-not-exist", "/api/v1/users/me"])
async def test_security_headers_are_set_on_success_and_error_responses(api_client, path):
    resp = await api_client.get(path)

    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"
    assert resp.headers["cache-control"] == "no-store"


async def test_secrets_never_appear_in_logs(api_client, sign_in, caplog):
    caplog.set_level(logging.INFO)
    login = await sign_in()
    access = login.json()["access_token"]
    refresh = login.cookies["refresh_token"]

    await api_client.get("/api/v1/users/me", headers={"Authorization": f"Bearer {access}"})
    await api_client.get("/api/v1/users/me", headers={"Authorization": "Bearer leaked.token.value"})
    await api_client.post("/api/v1/auth/refresh", headers={"Cookie": f"refresh_token={refresh}"})

    settings = get_settings()
    secrets_ = [
        access,
        refresh,
        "leaked.token.value",
        "auth-code",  # the OAuth code used by the sign_in fixture
        "test-client-secret",
        settings.jwt_secret_key.get_secret_value(),
        settings.session_secret_key.get_secret_value(),
        settings.openai_api_key.get_secret_value(),
    ]
    assert "/api/v1/auth/google/callback" in caplog.text  # logging was captured: not vacuous
    for secret in secrets_:
        assert secret not in caplog.text


async def test_unicode_filenames_are_preserved(upload, alice):
    resp = await upload(alice, filename="रिपोर्ट 📄.txt")

    assert resp.status_code == 202
    body = resp.json()
    assert body["original_filename"] == "रिपोर्ट 📄.txt"
    assert body["name"] == "रिपोर्ट 📄"


async def test_very_long_filenames_are_truncated_and_stored(upload, alice):
    resp = await upload(alice, filename="a" * 400 + ".txt")

    assert resp.status_code == 202
    body = resp.json()
    assert len(body["original_filename"]) == 255
    assert body["original_filename"].endswith(".txt")
    assert len(body["name"]) <= 255


async def test_control_and_bidi_characters_are_removed_from_the_document_name(upload, alice):
    resp = await upload(alice, name="Report\u202e\x00 final")

    assert resp.status_code == 202
    assert resp.json()["name"] == "Report final"


async def test_a_blank_name_falls_back_to_the_filename(upload, alice):
    resp = await upload(alice, filename="Budget 2026.txt", name="   ")

    assert resp.status_code == 202
    assert resp.json()["name"] == "Budget 2026"
