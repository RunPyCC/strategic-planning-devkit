# strategic-planning-devkit
- [ ] #TODO Decide whether to use or combine:
    - v1 Purpose: A printable, modular “decision kit” for mapping systems, dependencies, and access on a whiteboard (or any magnetic surface).
    - v2 Purpose: To facilitate whiteboard driven presentations through 3D printed icons representing digital architecture.

- [ ] #TODO Decide whether to use or combine:
    - v1 How it works: Digital architecture are commonly comprised of reused service icons. Take an icon (`.svg`) -> 3D dev tool (`.stl`).
    - v2 How it works: A printable, modular “decision kit” for mapping systems, dependencies, and access on a whiteboard (or any magnetic surface).

- [ ] #TODO Add a 'Why it matters':

This repo contains **parametric OpenSCAD models** that turn **SVG icons/logos** into physical **tiles**, plus **pockets** that hold tiles and optionally encode “levels of access” using magnet patterns—so you can make architecture and decision conversations more visual, faster.


## Table of Contents

## Installation 
Need to install OpenSCAD (2021).

### Linux Installations
1. [Configure flathub](https://flathub.org/en/setup)
2. Install OpenSCAD: 
```bash
flatpak install flathub org.openscad.OpenSCAD
```
3. Clone Respository

## Create a New Tile (.stl)
1. Run OpenSCAD:
```bash
flatpak run org.openscad.OpenSCAD
```
2. Click `Open` and select the `devkit/tile.scad`.
3. In the in-app editor, set `ICON_SVG` to the SVG you want to emboss/engrave on the tile (see the [icon selector block](devkit/tile.scad)). Example:

```scad
ICON_SVG = "icons/misc/digital-ocean.svg";
```
4. Adjust any parameters you want to tweak (see [Tile parameters](#tile-parameters)).
5. Click `Render` and preview the results.
6. Click `Export as STL` and save at the desired location.
7. Import the new `.stl` into your slicer software and print.

## Create a New Pocket (.stl)
1. Click `Open` and select the `devkit/pocket.scad`.
2. In the In-app editor uncomment the line of the `.svg` file you want to use.

```scad
ICON_SVG = "icons/misc/digital-ocean.svg";
```
3. Adjust any parameters you want to tweak (see [Pocket parameters](#pocket-parameters)).
4. Click `Render` and preview the results.
5. Click `Export as STL` and save at the desired location.
6. Import the new `.stl` into your slicer software and print.

---

## Tile parameters
The most common parameters for `devkit/tile.scad` live near the top of the file and can be edited directly in OpenSCAD:

| Parameter | What it does | Default (from `lib.scad`) |
| --- | --- | --- |
| `ICON_SVG` | Path to the SVG you want recessed/embossed. Uncomment an entry in the selector list or set your own. | `"icons/misc/docker.svg"` |
| `TILE_HEIGHT` | Overall tile thickness. | `TILE_H = 1.2` mm |
| `ICON_DEPTH` | Depth of the icon relief. | `ICON_H = 0.5` mm |
| `TILE_CLEARANCE` | Per-side clearance so the tile fits a matching pocket. Increase if your printer is tight. | `FIT = CLEARANCE/2 = 0.05` mm |
| `USE_SPINNER_HOLE` | Add/remove the finger/spinner hole. | `true` |
| `SPINNER_DIAMETER` | Spinner hole diameter. | `SPINNER_D = 20` mm |
| `TILE_D` | Overall tile width/diameter. Slightly smaller than the pocket (`COASTER_D - 2.4`). | `101.6 - 2.4` mm |

Tips:
- If your tiles bind in pockets, bump `TILE_CLEARANCE` up by `0.05–0.10` mm and re-render.
- Turn off `USE_SPINNER_HOLE` if you prefer a solid face for more icon area.

## Pocket parameters
Key parameters for `devkit/pocket.scad` are also defined at the top of the file:

| Parameter | What it does | Default (from `lib.scad`) |
| --- | --- | --- |
| `ICON_SVG` | Path to the SVG shown on the pocket face. | `"icons/misc/docker.svg"` |
| `POCKET_HEIGHT` | Overall pocket thickness. | `POCKET_H = 3.0` mm |
| `ICON_DEPTH` | Depth of the icon recess. | `ICON_H = 0.5` mm |
| `POCKET_CLEARANCE` | Per-side clearance so tiles slide in/out easily. | `FIT = 0.05` mm |
| `USE_SPINNER_HOLE` | Add/remove the spinner/carry hole. | `true` |
| `SPINNER_DIAMETER` | Spinner hole diameter. | `SPINNER_D = 20` mm |
| `USE_MAGNET_CUTOUTS` | Include magnet cavities on the base. Disable if using adhesives instead of embedded magnets. | `true` |
| `MAGNET_Z0` | Z-offset where magnet cavities start (controls bottom skin thickness). | `BOTTOM_SKIN = 0.3` mm |
| `USE_TILE_RETENTION` | Adds a small lip to keep the tile from wobbling (octagon only). | `false` |
| `RET_XY_CLEAR` | Side-to-side slack between the tile and retention lip. Lower = tighter. | `0.25` mm |
| `RET_WALL_THICK` | Thickness of the retention lip wall. | `2.0` mm |
| `RET_ANCHOR_Z` | How deep the retention lip sinks into the pocket top surface. | `0.6` mm |
| `RET_Z_DEPTH` | How far the retention lip extends upward. Should exceed tile thickness. | `TILE_H + 2` mm |
| `RET_DRAFT` | Small taper on the outer face of the retention lip. | `0.010` (scale factor) |

Tips:
- If magnets are loose, decrease `MAG_CLEAR` in `devkit/lib.scad` by `0.1–0.2` mm; if they do not fit, increase it.
- `USE_TILE_RETENTION` is off by default—enable it when you want pockets to hold tiles firmly on mobile boards.

---

Thank you to the OpenSCAD creators!

---

## What you can build

### Tiles (with icons)
- Import an SVG (cloud service, app, database, etc.)
- Generate a thin tile with a recessed/embossed icon
- Optional hole near the top for carry, hanging, or spinner-style attachments

### Pockets (tile holders)
- A fitted slot that holds tiles upright on a whiteboard
- Optional magnet cavities for magnetic mounting
- Optional magnet grid regions intended to represent “capabilities/access” (ex: read/query/write/admin)

### Case (in progress)
- A carry and storage solution for pockets + tiles
- Experimental add-ons like measurement / spacing guides and rotation tools

---

## Why this exists

Most architecture and operational decisions happen on a whiteboard, in a conference room, or in a workshop—often with people who don’t live in diagrams every day.

A physical kit helps:
- **Make system relationships tangible** (what talks to what, where boundaries are)
- **Reduce ambiguity** (everyone sees the same thing)
- **Speed up alignment** (less time explaining symbols, more time discussing tradeoffs)

---

## Quick start

### 1. Install prerequisites
- **OpenSCAD** (required)
- A slicer (Anycubic Slicer Next, PrusaSlicer, Cura, Bambu Studio, etc.)

### 2. Open a model
Open these in OpenSCAD:
- `devkit/icon.scad`
- `devkit/tile.scad`
- `devkit/pocket.scad`

### 3. Export STL
In OpenSCAD: `File` → `Export` → `Export as STL`

Or via CLI (example):

Example: tile.scad
```scad
openscad -o out/tile.stl devkit/tile.scad
```

Example: pocket.scad
```scad
openscad -o out/pocket.stl devkit/pocket.scad
```

## Using your own SVG icons
1. Add your `.svg` to the icons directory (or wherever you prefer)
2. Point the model at it (typically via an ICON_SVG variable/parameter in the OpenSCAD file)
3. Render → export STL

Tip: If your SVG has lots of empty space around the logo, crop it in Inkscape so the icon fills the canvas.

---

## Common customization knobs
You’ll see parameters like these in the .scad files:

- **Shape & size**

  - `SHAPE` (e.g., circle / polygon variants depending on the library)
  - `COASTER_D` / tile diameter / across-flats sizing
- **Thickness / depths**

  - `TILE_H`, `ICON_H` (tile height and icon depth)
- **Fit & clearance**

  - `FIT` or similar clearance values for tighter/looser pockets
- **Magnets**

  - `MAG_D`, `MAG_H`, `MAG_CLEAR` (diameter, thickness, clearance)
- **Optional features**

  - spinner hole on tiles / pockets (for carry or attachment systems)

---

## Example workflows

### “Architecture mapping” set

- Tiles: AWS services, databases, apps, queues
- Pockets: one per layer or environment
- Arrange on a whiteboard to tell the story: request flow, boundaries, dependencies

### “Access / permissions” set

- A pocket can include a magnet grid region intended to represent capabilities like:

  - ping/healthcheck, read, query, write, delete, admin, etc.
- Use magnet inlays as physical “bits” to show allowed actions

---
## Repository layout (high level)

- `devkit/`

  - `icon.scad` – icon handling / extrusion from SVG
  - `tile.scad` – tile geometry + features
  - `pocket.scad` – tile holder geometry + magnets
  - `lib.scad` – shared parameters + reusable modules
  - `icons/` – SVG icon library (optional / expandable)

---

## Roadmap

Planned enhancements (public-facing):

- **Case v1**: portable storage for tiles + pockets (in development)
- **More shapes**: additional outlines and better “safe area” rules for icons
- **Magnet grid presets**: standard layouts for “capabilities/access” patterns
- **Better SVG pipeline**: sizing normalization + optional indexing tooling
- **Print profiles**: recommended slicer presets for common printers/materials
- **Example kits**: reference sets (cloud architecture, app stack, database ops)

---

## Contributing

Contributions are welcome and are most helpful aroudn:

- new icon tiles (SVG additions + tested outputs)
- geometry improvements (fit, tolerances, printability)
- documentation and example kits

If you contribute logos, make sure you have the right to redistribute them.

---

## Trademark & logo note

Third-party logos/icons (AWS, Docker, etc.) are trademarks of their respective owners. This project is not affiliated with or endorsed by those companies.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
