from __future__ import annotations

import tarfile

from visionbench.storage import S3Archive


class Client:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def upload_file(self, filename, bucket, key):
        self.calls.append((filename, bucket, key))
        if self.fail:
            raise OSError("network down")
        with tarfile.open(filename) as archive:
            assert archive.getnames() == ["result.txt"]


def test_s3_archive_uploads_only_run_contents(tmp_path):
    run = tmp_path / "run-v1"
    run.mkdir()
    (run / "result.txt").write_text("done")
    (tmp_path / "outside.txt").write_text("do not archive")
    client = Client()
    result = S3Archive("bucket", "archive/runs", client=client).archive_run(run)
    assert result.uploaded is True
    assert result.key == f"archive/runs/{tmp_path.name}/run-v1.tar.gz"
    assert client.calls[0][1:] == ("bucket", result.key)


def test_s3_failure_is_reported_and_leaves_run(tmp_path):
    run = tmp_path / "run-v1"
    run.mkdir()
    output = run / "result.txt"
    output.write_text("done")
    result = S3Archive("bucket", client=Client(fail=True)).archive_run(run)
    assert not result.uploaded
    assert "network down" in result.error
    assert output.read_text() == "done"


def test_missing_directory_is_reported(tmp_path):
    result = S3Archive("bucket", client=Client()).archive_run(tmp_path / "missing")
    assert not result.uploaded
    assert "does not exist" in result.error
