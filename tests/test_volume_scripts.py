"""The pure parts of scripts/datasets: annotations to world LPS boxes, and resumable fetches."""

import hashlib
import http.server
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "datasets"))

import _volumes  # noqa: E402
import duke_breast_mri  # noqa: E402
import lidc_idri  # noqa: E402
import luna16  # noqa: E402

from opendiagnostics.data.volumes import VolumeManifest  # noqa: E402


def axial(z: float, x0: float = -100.0, y0: float = -50.0, spacing: float = 0.5) -> dict:
    return {"position": [x0, y0, z], "orientation": [1, 0, 0, 0, 1, 0], "spacing": [spacing] * 2}


def test_luna16_nodules_become_cubes_and_unsized_exclusions_get_10_mm():
    annotations = [
        {"seriesuid": "s1", "coordX": "10", "coordY": "-20", "coordZ": "30", "diameter_mm": "6"},
    ]
    excluded = [
        {"seriesuid": "s1", "coordX": "0", "coordY": "0", "coordZ": "0", "diameter_mm": "-1"},
    ]
    manifest = luna16.manifest({"s1": 3, "s2": 0}, {"s1": "LIDC-IDRI-0001"}, annotations, excluded)
    spec = VolumeManifest.model_validate(manifest)
    assert not spec.patient_ids_available
    first, second = spec.volumes
    assert (first.id, first.fold, second.id, second.patient_id) == ("s2", 0, "s1", "LIDC-IDRI-0001")
    assert second.boxes == [(7.0, -23.0, 27.0, 13.0, -17.0, 33.0)]
    assert second.ignore == [(-5.0, -5.0, -5.0, 5.0, 5.0, 5.0)]
    assert first.boxes == []


def test_luna16_reports_patients_whose_scans_span_folds():
    folds = {"a": 0, "b": 1, "c": 1}
    patients = {"a": "p1", "b": "p1", "c": "p2"}
    assert luna16.cross_fold_patients(folds, patients) == ["p1"]


LIDC_XML = """<?xml version="1.0"?>
<LidcReadMessage xmlns="http://www.nih.gov">
  <ResponseHeader><SeriesInstanceUid>1.2.3</SeriesInstanceUid></ResponseHeader>
  {sessions}
</LidcReadMessage>"""


def _session(reader: str, body: str) -> str:
    return (
        f"<readingSession><servicingRadiologistID>{reader}</servicingRadiologistID>"
        f"{body}</readingSession>"
    )


def _outline(sop: str, points: list[tuple[int, int]], inclusion: str = "TRUE") -> str:
    edges = "".join(
        f"<edgeMap><xCoord>{x}</xCoord><yCoord>{y}</yCoord></edgeMap>" for x, y in points
    )
    return f"<roi><imageSOP_UID>{sop}</imageSOP_UID><inclusion>{inclusion}</inclusion>{edges}</roi>"


SQUARE = [(10, 20), (12, 20), (12, 22), (10, 22)]


def _reads(min_readers: int) -> tuple[list, list]:
    nodule = f"<unblindedReadNodule>{_outline('s1', SQUARE)}{_outline('s2', SQUARE)}</unblindedReadNodule>"
    shifted = f"<unblindedReadNodule>{_outline('s2', [(11, 21), (13, 23)])}</unblindedReadNodule>"
    hole = _outline("s2", [(0, 0), (500, 500)], inclusion="FALSE")
    small = f"<unblindedReadNodule>{_outline('s0', [(100, 100)])}{hole}</unblindedReadNodule>"
    non_nodule = (
        "<nonNodule><imageSOP_UID>s0</imageSOP_UID>"
        "<locus><xCoord>200</xCoord><yCoord>200</yCoord></locus></nonNodule>"
    )
    xml = LIDC_XML.format(
        sessions=_session("r1", nodule + small) + _session("r2", shifted + non_nodule)
    )
    series, marks = lidc_idri.parse_reads(xml)
    assert series == "1.2.3"
    assert [m.kind for m in marks] == ["nodule", "small", "nodule", "non_nodule"]
    geometry = {"s0": axial(0.0), "s1": axial(1.0), "s2": axial(2.0)}
    return lidc_idri.series_boxes(marks, geometry, min_readers)


def test_lidc_outline_box_covers_whole_pixels_and_half_a_slice():
    box = _volumes.pixel_box([(axial(1.0), 10, 20), (axial(2.0), 12, 22)], gap=1.0)
    # x = -100 + 0.5 * (10 - 0.5) to -100 + 0.5 * (12 + 0.5); z from 1 - 0.5 to 2 + 0.5.
    assert box == pytest.approx([-95.25, -40.25, 0.5, -93.75, -38.75, 2.5])


def test_lidc_overlapping_marks_of_two_readers_are_one_nodule_with_a_median_box():
    targets, ignore = _reads(min_readers=2)
    first = [-95.25, -40.25, 0.5, -93.75, -38.75, 2.5]
    second = [-94.75, -39.75, 1.5, -93.25, -38.25, 2.5]
    assert targets == [pytest.approx([(a + b) / 2 for a, b in zip(first, second, strict=True)])]
    centers = [[(b[k] + b[k + 3]) / 2 for k in range(3)] for b in ignore]
    assert centers == [pytest.approx([-50.0, 0.0, 0.0]), pytest.approx([0.0, 50.0, 0.0])]
    assert all(b[3] - b[0] == pytest.approx(10.0) for b in ignore)


def test_lidc_nodules_below_the_reader_threshold_are_ignored():
    targets, ignore = _reads(min_readers=3)
    assert targets == []
    assert len(ignore) == 3


def _series(description: str, uid: str) -> dict:
    return {"SeriesDescription": description, "SeriesInstanceUID": uid}


@pytest.mark.parametrize(
    ("descriptions", "pre", "post"),
    [
        (["ax t1", "ax dyn pre", "ax dyn 1st pass", "ax dyn 2nd pass"], 1, 2),
        (["ax 3d dyn", "Ph1/ax 3d dyn", "Ph2/ax 3d dyn", "Ax 3D T1 NON FS"], 0, 1),
        (["ax 3d pre", "1st ax dyn", "2nd ax dyn", "Segmentation"], 0, 1),
    ],
)
def test_duke_picks_the_pre_and_first_post_contrast_series(descriptions, pre, post):
    series = [_series(d, str(n)) for n, d in enumerate(descriptions)]
    assert duke_breast_mri.choose_series(series) == (series[pre], series[post])


def test_duke_explains_a_patient_without_a_clear_pair():
    series = [_series("ax dyn", "0"), _series("ax dyn", "1"), _series("ax t1", "2")]
    assert "0 first post-contrast" in duke_breast_mri.choose_series(series)


def test_duke_slices_count_from_1_in_ascending_height():
    geometry = {f"sop{n}": axial(10.0 - 2 * n, spacing=1.0) for n in range(5)}
    row = dict(zip(duke_breast_mri.BOX_COLUMNS, (21, 30, 11, 15, 2, 3), strict=True))
    box = duke_breast_mri.tumor_box(row, geometry)
    # Heights ascend 2, 4, 6, ...: slices 2 and 3 sit at z = 4 and 6. Columns 11..15 are
    # 0-based 10..14, rows 21..30 are 20..29, each widened by half a pixel.
    assert box == pytest.approx([-90.5, -30.5, 3.0, -85.5, -20.5, 7.0])


def test_duke_grids_must_match_to_stack_as_channels():
    a = {"x": axial(0.0), "y": axial(1.0)}
    assert duke_breast_mri.same_grid(a, {"p": axial(1.0), "q": axial(0.0)})
    assert not duke_breast_mri.same_grid(a, {"p": axial(0.0), "q": axial(1.5)})


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = Path(self.translate_path(self.path)).read_bytes()
        start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-") or 0)
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        self.wfile.write(body[start:])


@pytest.fixture
def served(tmp_path):
    source = tmp_path / "served"
    source.mkdir()
    (source / "file.bin").write_bytes(bytes(range(256)) * 40)

    def handler(*args, **kwargs):
        return _RangeHandler(*args, directory=str(source), **kwargs)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/file.bin", (source / "file.bin").read_bytes()
    server.shutdown()


def test_download_resumes_a_partial_file_and_checks_its_md5(served, tmp_path):
    url, body = served
    dest = tmp_path / "out" / "file.bin"
    dest.parent.mkdir()
    dest.with_name("file.bin.part").write_bytes(body[:1000])
    md5 = hashlib.md5(body).hexdigest()
    assert _volumes.download(url, dest, md5=md5, size=len(body)).read_bytes() == body
    assert not dest.with_name("file.bin.part").exists()


def test_a_download_with_the_wrong_md5_is_deleted(served, tmp_path):
    url, _ = served
    dest = tmp_path / "file.bin"
    with pytest.raises(OSError, match="does not match"):
        _volumes.download(url, dest, md5="0" * 32)
    assert not dest.exists()
    assert not dest.with_name("file.bin.part").exists()


def test_extract_refuses_members_outside_the_folder(tmp_path):
    import zipfile

    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "x")
    with pytest.raises(OSError, match="outside the folder"):
        _volumes.extract(archive, tmp_path / "out")
