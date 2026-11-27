import json

import numpy as np
import pytest
import torch
from conftest import fit_smoke
from torch import nn

from opendiagnostics.data import (
    Camelyon16DataModule,
    Camelyon17DataModule,
    SyntheticSlideDataModule,
)
from opendiagnostics.data.camelyon import camelyon16_slides, camelyon17_slides
from opendiagnostics.data.slides import level_for_downsample, pool, tile_grid
from opendiagnostics.data.synthetic import write_pyramid

C16_REFERENCE = """image,type,class,center
normal_001.tif,normal,negative,0
normal_002.tif,normal,negative,1
tumor_001.tif,tumor,macro,0
tumor_002.tif,tumor,micro,1
test_001.tif,tumor,macro,1
test_003.tif,normal,negative,0
"""

C17_STAGES = """patient,stage,center
patient_000.zip,pN0,0
patient_000_node_0.tif,negative,0
patient_000_node_1.tif,negative,0
patient_001.zip,pN1mi,1
patient_001_node_0.tif,micro,1
patient_001_node_1.tif,itc,1
patient_002.zip,pN2,2
patient_002_node_0.tif,macro,2
"""


def test_a_full_mask_keeps_every_whole_tile_and_drops_the_ragged_edge():
    coords = tile_grid(np.ones((10, 15), bool), (100, 150), 40, 0.5)
    assert coords.tolist() == [[0, 0], [0, 40], [0, 80], [40, 0], [40, 40], [40, 80]]


def test_an_empty_mask_keeps_no_tiles():
    assert tile_grid(np.zeros((10, 10), bool), (100, 100), 20, 0.1).shape == (0, 2)


def test_tiles_are_kept_by_their_tissue_fraction():
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True
    mask[2:4, 2:4] = True
    mask[2, 0] = mask[3, 0] = True
    coords = tile_grid(mask, (40, 40), 20, 0.5)
    assert coords.tolist() == [[20, 0], [20, 20]]
    assert tile_grid(mask, (40, 40), 20, 0.25).tolist() == [[0, 0], [20, 0], [20, 20]]


def test_tiles_smaller_than_a_mask_pixel_take_the_pixel_they_fall_in():
    mask = np.array([[True, False], [False, False]])
    coords = tile_grid(mask, (100, 100), 10, 1.0)
    assert len(coords) == 25
    assert coords.max() == 40


def test_the_mask_level_is_the_closest_downsample_on_a_log_scale():
    ratios = [1, 2, 4, 8, 16, 32, 64]
    assert level_for_downsample(ratios, 32) == 5
    assert level_for_downsample(ratios, 40) == 5
    assert level_for_downsample(ratios, 1000) == 6


def test_pool_turns_maps_and_lists_into_one_vector_per_tile():
    assert pool(torch.ones(2, 8)).shape == (2, 8)
    assert pool(torch.ones(2, 8, 3, 3)).shape == (2, 8)
    assert pool([torch.ones(2, 4, 6, 6), torch.ones(2, 8, 3, 3)]).shape == (2, 8)
    with pytest.raises(ValueError, match="num_classes: 0"):
        pool(torch.ones(2, 8, 3))


def test_camelyon16_uses_the_official_test_slides_and_tumor_as_positive(tmp_path):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "reference.csv").write_text(C16_REFERENCE)
    slides = camelyon16_slides(tmp_path)
    assert [(s.slide_id, s.label, s.split) for s in slides] == [
        ("normal_001", 0, "train"),
        ("normal_002", 0, "train"),
        ("tumor_001", 1, "train"),
        ("tumor_002", 1, "train"),
        ("test_001", 1, "test"),
        ("test_003", 0, "test"),
    ]
    assert slides[0].path == tmp_path / "images" / "normal_001.tif"


def test_camelyon17_labels_slides_and_stratifies_by_the_patients_stage(tmp_path):
    (tmp_path / "stages.csv").write_text(C17_STAGES)
    slides = camelyon17_slides(tmp_path)
    assert [(s.slide_id, s.patient_id, s.label, s.stratum) for s in slides] == [
        ("patient_000_node_0", "patient_000", 0, "pN0"),
        ("patient_000_node_1", "patient_000", 0, "pN0"),
        ("patient_001_node_0", "patient_001", 2, "pN1mi"),
        ("patient_001_node_1", "patient_001", 1, "pN1mi"),
        ("patient_002_node_0", "patient_002", 3, "pN2"),
    ]
    assert all(s.split is None for s in slides)


def _fake_features(directory, names, dim=4):
    directory.mkdir(parents=True)
    for n, name in enumerate(names):
        torch.save(
            {
                "features": torch.full((n + 1, dim), float(n)).half(),
                "coords": torch.zeros(n + 1, 2),
            },
            directory / f"{name}.pt",
        )


def test_camelyon16_carves_validation_from_train_and_keeps_test(tmp_path):
    (tmp_path / "evaluation").mkdir()
    rows = ["image,type,class,center"]
    rows += [f"normal_{n:03d}.tif,normal,negative,0" for n in range(20)]
    rows += [f"tumor_{n:03d}.tif,tumor,macro,0" for n in range(20)]
    rows += ["test_001.tif,tumor,macro,1", "test_002.tif,normal,negative,1"]
    (tmp_path / "evaluation" / "reference.csv").write_text("\n".join(rows) + "\n")
    names = [r.split(".tif")[0] for r in rows[1:]]
    _fake_features(tmp_path / "features", names)
    data = Camelyon16DataModule(
        root=str(tmp_path),
        features_dir=str(tmp_path / "features"),
        encoder=nn.Linear(2, 2),
        download=False,
    )
    data.setup()
    ids = {k: {data.source.sample_ids[i] for i in v} for k, v in data.splits.items()}
    assert ids["test"] == {"test_001", "test_002"}
    assert len(ids["val"]) == 6
    assert not ids["val"] & ids["train"]
    assert {i[:5] for i in ids["val"]} == {"norma", "tumor"}
    bag, label = data.source.dataset[names.index("test_001")]
    assert bag.dtype == torch.float32
    assert bag.shape == (41, 4)
    assert label == 1
    batch = next(iter(data.test_dataloader()))
    assert batch["image"].shape == (1, 41, 4)


def test_features_from_another_encoder_are_refused(tmp_path):
    (tmp_path / "stages.csv").write_text(C17_STAGES)
    _fake_features(tmp_path / "features", [])
    kwargs = {"root": str(tmp_path), "features_dir": str(tmp_path / "features")}
    Camelyon17DataModule(encoder=nn.Linear(2, 2), **kwargs)._check_record({"encoder_sha256": "a"})
    with pytest.raises(ValueError, match=r"other settings \(encoder_sha256\)"):
        Camelyon17DataModule(encoder=nn.Linear(2, 2), **kwargs)._check_record(
            {"encoder_sha256": "b"}
        )


def test_missing_slides_point_at_the_download_scripts(tmp_path):
    (tmp_path / "stages.csv").write_text(C17_STAGES)
    data = Camelyon17DataModule(
        root=str(tmp_path),
        features_dir=str(tmp_path / "features"),
        encoder=nn.Linear(2, 2),
        download=False,
    )
    with pytest.raises(FileNotFoundError, match="scripts/datasets"):
        data.prepare_data()


def test_slide_batches_must_hold_one_bag(tmp_path):
    with pytest.raises(ValueError, match="batch_size must be 1"):
        Camelyon17DataModule(
            root=str(tmp_path), features_dir=str(tmp_path), encoder=nn.Identity(), batch_size=4
        )


class MeanColour(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean((2, 3))


def test_extraction_tiles_only_tissue_and_resumes(tmp_path):
    data = SyntheticSlideDataModule(
        root=str(tmp_path),
        features_dir=str(tmp_path / "features"),
        encoder=MeanColour(),
        num_slides=3,
        slide_size=256,
        patch_size=16,
        patch_level=1,
        mask_downsample=4,
        extract_device="cpu",
        extract_workers=0,
    )
    data.prepare_data()
    files = sorted((tmp_path / "features").glob("*.pt"))
    assert len(files) == 3
    saved = torch.load(files[0], weights_only=True)
    assert saved["features"].shape == (len(saved["coords"]), 3)
    assert saved["features"].dtype == torch.float16
    background = torch.tensor([240 / 255] * 3)
    assert not torch.isclose(saved["features"].float(), background, atol=0.02).all(1).any()
    stamps = {f.name: f.stat().st_mtime_ns for f in files}
    files[1].unlink()
    data.prepare_data()
    assert {f.name: f.stat().st_mtime_ns for f in files if f != files[1]} == {
        k: v for k, v in stamps.items() if k != files[1].name
    }
    assert files[1].exists()


def test_openslide_reads_the_generated_pyramid(tmp_path):
    from monai.data import WSIReader

    image = np.zeros((128, 96, 3), np.uint8)
    image[:, :, 0] = 200
    write_pyramid(tmp_path / "s.tif", image, levels=3)
    reader = WSIReader(backend="openslide")
    wsi = reader.read(str(tmp_path / "s.tif"))
    assert reader.get_level_count(wsi) == 3
    assert reader.get_downsample_ratio(wsi, 2) == 4
    tile, _ = reader.get_data(wsi, location=(64, 32), size=(8, 8), level=1)
    assert tile.shape == (3, 8, 8)
    assert (tile[0] == 200).all()


def test_slide_smoke_config_trains_end_to_end(tmp_path):
    slides = tmp_path / "slides"
    run_dir = fit_smoke(
        "slide_classification",
        tmp_path,
        f"--data.init_args.root={slides}",
        f"--data.init_args.features_dir={slides / 'features'}",
    )
    metrics = json.loads((run_dir / "metrics.json").read_text())
    test = metrics["test"]["auroc"]
    assert test["lower"] <= test["value"] <= test["upper"]
    assert test["unit"] == "patient"
    assert json.loads((run_dir / "examples" / "examples.json").read_text()) == []
    with (run_dir / "predictions_test.csv").open() as handle:
        assert handle.readline().strip() == "id,patient_id,target,prob/tumor"


def _settings(**changes):
    from dataclasses import replace

    from opendiagnostics.data.slides import Extraction

    return replace(Extraction(16, 1, 4, 0.25, "openslide", 8, 0, "cpu"), **changes)


def test_a_blank_slide_and_a_missing_level_are_errors_that_say_what_to_change(tmp_path):
    from opendiagnostics.data.slides import extract_slide

    write_pyramid(tmp_path / "blank.tif", np.full((128, 128, 3), 240, np.uint8), levels=3)
    with pytest.raises(ValueError, match="lower min_tissue"):
        extract_slide(tmp_path / "blank.tif", _settings(), MeanColour(), None)
    with pytest.raises(ValueError, match="patch_level 5 does not exist"):
        extract_slide(tmp_path / "blank.tif", _settings(patch_level=5), MeanColour(), None)


def test_the_encoder_sees_transformed_tiles_and_tiles_survive_pickling(tmp_path):
    import pickle

    from opendiagnostics.data.slides import SlideTiles, extract_slide

    image = np.full((128, 128, 3), 240, np.uint8)
    image[16:112, 16:112] = (200, 100, 150)
    write_pyramid(tmp_path / "s.tif", image, levels=3)
    saved = extract_slide(tmp_path / "s.tif", _settings(), MeanColour(), lambda x: x * 0 + 7)
    assert (saved["features"] == 7).all()
    tiles = SlideTiles(tmp_path / "s.tif", "openslide", saved["coords"].numpy(), 16, 1)
    first = tiles[0]
    assert torch.equal(pickle.loads(pickle.dumps(tiles))[0], first)
