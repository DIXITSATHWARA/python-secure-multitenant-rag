from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.services.storage import LocalStorage, S3Storage, build_storage


class FakeClientError(Exception):
    """Shaped like botocore's ClientError: carries a .response dict."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeS3:
    """Just enough of boto3's S3 client for S3Storage."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, datetime]] = {}

    def put_object(self, *, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = (Body, datetime.now(UTC))

    def get_object(self, *, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise FakeClientError("NoSuchKey")
        body = self.objects[Key][0]
        return {"Body": SimpleNamespace(read=lambda: body)}

    def delete_object(self, *, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, *, Bucket, Prefix):  # noqa: N803
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        if not keys:
            yield {}  # a real empty bucket has no "Contents" key
            return
        for part in (keys[:1], keys[1:]):  # two pages, to exercise pagination
            yield {"Contents": [{"Key": k, "LastModified": self.objects[k][1]} for k in part]}


async def test_local_list_keys_returns_relative_keys_and_timestamps(tmp_path):
    storage = LocalStorage(tmp_path / "s")
    await storage.save("a/b.txt", b"1")
    await storage.save("c.pdf", b"2")

    objects = await storage.list_keys()

    assert sorted(o.key for o in objects) == ["a/b.txt", "c.pdf"]
    for obj in objects:
        assert obj.modified.tzinfo is not None
        assert abs((datetime.now(UTC) - obj.modified).total_seconds()) < 60


async def test_local_list_keys_includes_leftover_tmp_files(tmp_path):
    storage = LocalStorage(tmp_path / "s")
    (storage.root / "u").mkdir()
    (storage.root / "u" / "d.txt.tmp").write_bytes(b"x")

    assert [o.key for o in await storage.list_keys()] == ["u/d.txt.tmp"]


async def test_s3_round_trip_applies_the_prefix():
    s3 = FakeS3()
    storage = S3Storage(s3, "bucket", "/tenant-a/")

    await storage.save("u/d.txt", b"hello")

    assert "tenant-a/u/d.txt" in s3.objects
    assert await storage.read("u/d.txt") == b"hello"


async def test_s3_read_of_a_missing_key_is_file_not_found():
    with pytest.raises(FileNotFoundError):
        await S3Storage(FakeS3(), "bucket").read("missing.txt")


async def test_s3_read_errors_other_than_not_found_propagate():
    s3 = FakeS3()

    def denied(**kwargs):
        raise FakeClientError("AccessDenied")

    s3.get_object = denied

    with pytest.raises(FakeClientError):
        await S3Storage(s3, "bucket").read("k")


async def test_s3_delete_is_idempotent():
    s3 = FakeS3()
    storage = S3Storage(s3, "bucket")
    await storage.save("k.txt", b"x")

    await storage.delete("k.txt")
    await storage.delete("k.txt")  # already gone: not an error

    assert s3.objects == {}


async def test_s3_list_keys_strips_the_prefix_and_ignores_other_keys():
    s3 = FakeS3()
    storage = S3Storage(s3, "bucket", "tenant-a")
    await storage.save("x/y.txt", b"1")
    await storage.save("x/z.txt", b"2")
    s3.objects["other/w.txt"] = (b"3", datetime.now(UTC))

    objects = await storage.list_keys()

    assert sorted(o.key for o in objects) == ["x/y.txt", "x/z.txt"]
    assert all(o.modified.tzinfo is not None for o in objects)


async def test_s3_list_keys_on_an_empty_bucket():
    assert await S3Storage(FakeS3(), "bucket").list_keys() == []


def test_unsupported_backend_is_rejected():
    settings = get_settings().model_copy(update={"storage_backend": "ftp"})

    with pytest.raises(RuntimeError, match="Unsupported"):
        build_storage(settings)
