import importlib
import io
import sys
import urllib.request

import numpy as np
import pytest
from fake_dicom import write_dicom

from opendiagnostics.data import BoxManifestDataModule


@pytest.fixture(scope="module")
def scripts(root):
    sys.path.insert(0, str(root / "scripts" / "datasets"))
    try:
        yield {
            name: importlib.import_module(name)
            for name in ("common", "rsna_pneumonia", "cbis_ddsm")
        }
    finally:
        sys.path.remove(str(root / "scripts" / "datasets"))


class Response(io.BytesIO):
    def __init__(self, body: bytes, status: int) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}


def test_download_resumes_a_partial_file_with_a_range_request(scripts, tmp_path):
    body = b"0123456789"
    dest = tmp_path / "file.zip"
    (tmp_path / "file.zip.part").write_bytes(body[:4])
    sent = []

    def opener(request: urllib.request.Request, timeout: int) -> Response:
        sent.append(request.get_header("Range"))
        return Response(body[4:], 206)

    scripts["common"].download("https://example.org/file.zip", dest, opener)
    assert sent == ["bytes=4-"]
    assert dest.read_bytes() == body
    assert not (tmp_path / "file.zip.part").exists()
    scripts["common"].download("https://example.org/file.zip", dest, opener)
    assert len(sent) == 1


def test_download_restarts_when_the_server_ignores_the_range(scripts, tmp_path):
    (tmp_path / "f.part").write_bytes(b"stale")
    scripts["common"].download(
        "https://example.org/f", tmp_path / "f", lambda request, timeout: Response(b"fresh", 200)
    )
    assert (tmp_path / "f").read_bytes() == b"fresh"


def test_rsna_manifest_maps_images_to_nih_patients_and_keeps_negatives(scripts, tmp_path):
    rsna = scripts["rsna_pneumonia"]
    pixels = np.zeros((8, 8), np.uint16)
    write_dicom(tmp_path / "images" / "a.dcm", pixels, SOPInstanceUID="1.2.1", PatientID="uuid-a")
    write_dicom(tmp_path / "images" / "b.dcm", pixels, SOPInstanceUID="1.2.2", PatientID="uuid-b")
    write_dicom(tmp_path / "images" / "c.dcm", pixels, SOPInstanceUID="1.2.3", PatientID="uuid-c")
    write_dicom(tmp_path / "images" / "unlabelled.dcm", pixels, PatientID="uuid-d")
    box = {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}
    export = {
        "datasets": [
            {
                "name": "Train",
                "annotations": [
                    {"SOPInstanceUID": "1.2.1", "labelId": "L1", "data": box},
                    {"SOPInstanceUID": "1.2.1", "labelId": "L1", "data": box},
                    {"SOPInstanceUID": "1.2.2", "labelId": "L2", "data": None},
                    {"SOPInstanceUID": "9.9.9", "labelId": "L2", "data": None},
                ],
            },
            {"name": "Test", "annotations": [{"SOPInstanceUID": "1.2.3", "labelId": "L2"}]},
        ]
    }
    mapping = [
        {"rsna": "uuid-a", "nih": "00000013_005.png"},
        {"rsna": "uuid-b", "nih": "00000013_006.png"},
        {"rsna": "uuid-c", "nih": "00000099_000.png"},
    ]
    index = rsna.index_dicoms(tmp_path / "images")
    boxes = rsna.annotated_boxes(export, index)
    rows = rsna.manifest_rows(boxes, rsna.nih_patients(mapping, index), tmp_path)
    assert [(r["path"], r["patient_id"], r["label"], r["x2"], r["y2"]) for r in rows] == [
        ("images/a.dcm", "00000013", "opacity", 4.0, 6.0),
        ("images/b.dcm", "00000013", "", "", ""),
        ("images/c.dcm", "00000099", "", "", ""),
    ]


def test_rsna_refuses_an_image_the_mapping_does_not_cover(scripts, tmp_path):
    rsna = scripts["rsna_pneumonia"]
    write_dicom(tmp_path / "a.dcm", np.zeros((4, 4), np.uint16), SOPInstanceUID="1.2.1")
    index = rsna.index_dicoms(tmp_path)
    boxes = rsna.annotated_boxes(
        {"datasets": [{"annotations": [{"SOPInstanceUID": "1.2.1"}]}]}, index
    )
    with pytest.raises(SystemExit, match="1 of 1 annotated images have no NIH patient ID"):
        rsna.manifest_rows(boxes, rsna.nih_patients({"other": "00000001_000.png"}, index), tmp_path)


CASE_HEADER = (
    "patient_id,breast_density,left or right breast,image view,abnormality id,abnormality type,"
    "pathology,image file path,cropped image file path,ROI mask file path\n"
)


def _case(subject, participant, side, view, n, pathology="MALIGNANT"):
    return (
        f"{participant},3,{side},{view},{n},x,{pathology},{subject}/1.2/1.3/000000.dcm,"
        f'{subject}_{n}/1.4/1.5/000000.dcm,"{subject}_{n}/1.4/1.5/000001.dcm\n"\n'
    )


def _mammogram(dicom, subject, pixels):
    write_dicom(dicom / subject / "s" / "000000.dcm", pixels)


def _roi(dicom, subject, shape, box):
    x1, y1, x2, y2 = box
    mask = np.zeros(shape, np.uint16)
    mask[y1:y2, x1:x2] = 255
    write_dicom(dicom / subject / "s" / "000000.dcm", np.full((3, 3), 7, np.uint16))
    write_dicom(dicom / subject / "s" / "000001.dcm", mask)


def test_cbis_manifest_boxes_masks_merges_one_film_and_keeps_test_whole(scripts, tmp_path):
    cbis = scripts["cbis_ddsm"]
    root, dicom = tmp_path, tmp_path / "dicom"
    film = np.arange(12 * 10, dtype=np.uint16).reshape(12, 10)
    (tmp_path / "mass_train.csv").write_text(
        CASE_HEADER
        + _case("Mass-Training_P_00001_LEFT_CC", "P_00001", "LEFT", "CC", 1)
        + _case("Mass-Training_P_00002_LEFT_CC", "P_00002", "LEFT", "CC", 1)
    )
    (tmp_path / "calc_train.csv").write_text(
        CASE_HEADER + _case("Calc-Training_P_00001_LEFT_CC", "P_00001", "LEFT", "CC", 1)
    )
    (tmp_path / "calc_test.csv").write_text(
        CASE_HEADER + _case("Calc-Test_P_00002_RIGHT_MLO", "P_00002", "RIGHT", "MLO", 1)
    )
    _mammogram(dicom, "Mass-Training_P_00001_LEFT_CC", film)
    _mammogram(dicom, "Calc-Training_P_00001_LEFT_CC", film)
    _mammogram(dicom, "Mass-Training_P_00002_LEFT_CC", film)
    _mammogram(dicom, "Calc-Test_P_00002_RIGHT_MLO", film)
    _roi(dicom, "Mass-Training_P_00001_LEFT_CC_1", (12, 10), (2, 3, 6, 9))
    _roi(dicom, "Calc-Training_P_00001_LEFT_CC_1", (12, 10), (7, 0, 8, 1))
    _roi(dicom, "Mass-Training_P_00002_LEFT_CC_1", (12, 10), (0, 0, 4, 4))
    _roi(dicom, "Calc-Test_P_00002_RIGHT_MLO_1", (12, 10), (1, 1, 3, 3))
    lesions = [
        *cbis.read_cases(tmp_path / "mass_train.csv", "mass", "train"),
        *cbis.read_cases(tmp_path / "calc_train.csv", "calcification", "train"),
        *cbis.read_cases(tmp_path / "calc_test.csv", "calcification", "test"),
    ]
    assert lesions[0].roi_subject == "Mass-Training_P_00001_LEFT_CC_1"
    images, skipped = cbis.mammograms(lesions, dicom)
    assert skipped["duplicate film merged"] == 1
    assert cbis.drop_test_participants_from_train(images) == 1
    rows = cbis.manifest_rows(images, root)
    assert [
        (r["patient_id"], r["split"], r["label"], r["x1"], r["y1"], r["x2"], r["y2"]) for r in rows
    ] == [
        ("P_00002", "test", "calcification", 1.0, 1.0, 3.0, 3.0),
        ("P_00001", "train", "mass", 2.0, 3.0, 6.0, 9.0),
        ("P_00001", "train", "calcification", 7.0, 0.0, 8.0, 1.0),
    ]
    manifest = scripts["common"].write_manifest(tmp_path / "manifest.csv", rows)
    data = BoxManifestDataModule(manifest=str(manifest), split_column="split")
    assert data.info.class_names == ("calcification", "mass")


def test_cbis_fetches_only_mask_series_when_tcia_lists_them_apart(scripts):
    cbis = scripts["cbis_ddsm"]
    lesion = cbis.Lesion("mass", "train", "P_00001", "LEFT", "CC", "BENIGN", "Mass_P", "Mass_P_1")
    series = [
        {
            "PatientID": "Mass_P",
            "SeriesInstanceUID": "full",
            "SeriesDescription": "full mammogram images",
        },
        {
            "PatientID": "Mass_P_1",
            "SeriesInstanceUID": "crop",
            "SeriesDescription": "cropped images",
        },
        {
            "PatientID": "Mass_P_1",
            "SeriesInstanceUID": "mask",
            "SeriesDescription": "ROI mask images",
        },
        {
            "PatientID": "Other",
            "SeriesInstanceUID": "other",
            "SeriesDescription": "full mammogram images",
        },
    ]
    assert cbis.series_to_fetch([lesion], series) == {"Mass_P": ["full"], "Mass_P_1": ["mask"]}


def test_cbis_series_download_unpacks_each_zip_once(scripts, tmp_path):
    import zipfile

    cbis = scripts["cbis_ddsm"]
    fetched = []

    def fetch(url, dest):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as handle:
            handle.writestr("1-1.dcm", b"dicom")
        return dest

    for _ in range(2):
        cbis.fetch_series({"Mass_P": ["9.9"]}, tmp_path, 2, True, fetch)
    assert fetched == [f"{cbis.NBIA}getImage?SeriesInstanceUID=9.9"]
    assert (tmp_path / "dicom" / "Mass_P" / "9.9" / "1-1.dcm").read_bytes() == b"dicom"
    assert not (tmp_path / "archives" / "9.9.zip").exists()
