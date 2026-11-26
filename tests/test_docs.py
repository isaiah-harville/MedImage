import re


def test_every_module_has_a_code_reference_page(root):
    referenced = set()
    for page in (root / "docs" / "reference").glob("*.md"):
        referenced |= set(re.findall(r"^::: (\S+)$", page.read_text(), flags=re.MULTILINE))
    modules = set()
    for path in (root / "src" / "opendiagnostics").rglob("*.py"):
        parts = path.relative_to(root / "src").with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if name in {
            "opendiagnostics",
            "opendiagnostics.data",
            "opendiagnostics.metrics",
            "opendiagnostics.modules",
        }:
            continue
        if parts[-1].startswith("_"):
            continue
        modules.add(name)
    missing = sorted(modules - referenced)
    assert not missing, f"add these modules to a page in docs/reference/: {missing}"


def test_every_hand_written_page_is_in_the_site_nav(root):
    import tomllib

    nav = tomllib.loads((root / "zensical.toml").read_text())["project"]["nav"]

    def pages(entries):
        for entry in entries:
            if isinstance(entry, str):
                yield entry
            else:
                for value in entry.values():
                    yield from pages(value) if isinstance(value, list) else [value]

    listed = set(pages(nav))
    docs = root / "docs"
    written = {
        str(p.relative_to(docs))
        for p in docs.rglob("*.md")
        if p.relative_to(docs).parts[0] != "notebook"
    }
    assert written - listed == set(), "add these pages to the nav in zensical.toml"
