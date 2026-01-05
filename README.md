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
3. In the In-app editor uncomment the line of the `.svg` file you want to use.

```scad
ICON_SVG = "icons/misc/digital-ocean.svg"
```
4. Click `Render` & preview the results
5. Click `Export as STL` save at desired location. 
6. Import the new `.stl` into your slicer software & print. 

## Create a New Pocket (.stl)
1. Click `Open` and select the `devkit/pocket.scad`.
2. In the In-app editor uncomment the line of the `.svg` file you want to use.

```scad
ICON_SVG = "icons/misc/digital-ocean.svg"
```
3. Click `Render` & preview the results
4. Click `Export as STL` save at desired location. 
5. Import the new `.stl` into your slicer software & print. 

Parameter list
USE_MAGNET_CUTOUTS
USE_SPINNER_HOLE
USE_TILE_RETENTION = True //

---

Thank you to the OpenSCAD creators!

---
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
- Use magnets (or empty cavities) as physical “bits” to show allowed actions

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

Contributions are welcome—especially:

- new icon tiles (SVG additions + tested outputs)
- geometry improvements (fit, tolerances, printability)
- documentation and example kits

If you contribute logos, make sure you have the right to redistribute them.

---

## Trademark & logo note

Third-party logos/icons (AWS, Docker, etc.) are trademarks of their respective owners. This project is not affiliated with or endorsed by those companies.

---

## License

Add your chosen license here (MIT/Apache-2.0/etc.).
 m 

---

Tiles Parameters:
- 

Pocket Parameters: 
- Magnet inserts
- Retaining wall


---


- [Marketing Messaging](#marketing-messaging-/-customer-value-proposition)
    - [Design Architecture](#design-architecture)
    - [Features](#features)
    - [Components](#pieces)
- [ARCHIVE TODO - COMPLETED](#archive-todo---completed)
- [References](#references)

---

### Design Architecture
Uses (4) major components at this point:
1. `icon`: The icon that will go onto the tile which includes things like Docker, nginx, PostgreSQL, and Cloud Services like AWS ECS.
2. `tile`: This is a thin tile that displays the *icon*. 
3. `pocket`: A magnetic component designed to display tiles for Digital Architectures & with a magnetic whiteboard in mind. 
4. `case`: Acts as a carry case for the remaining `pocket` & `tile` components. 

### Features
1. `icon`: See above.
2. `tile`: See above.
3. `pocket`: Provides a way for a user to visually show how much access one system, VM or container should have to another.
    - *Example*: (2) 4x2 grids for each side (left, right and bottom) section could allow you to show whether the system can:
        - Ping the other system to see if it is operational
        - Read data on the system
        - Query data
        - Modify data
        - Delete data
        - Admin access
        - etc. (possibly not in that order but it should demonstrate the point). 
4. `case`: 
 - Common Whiteboard Tools
    - Coding spacing & indent guide:
        - Solves writing code with inaccurate spacing & indents. 
    - Protractor:
        - Solves the user not being able to draw a perfect circle & .

### Components
1. `icon`: Provides the framework for the icon that will be put onto a tile. 
2. `tile`: 
    - [ ] Displays the icon in a way that can fit into a pocket or pocket slot.
    - [ ] Has feet and holds to be are stackable on top of each other creating a type of collapse experience if the user has 6 tiles that should all be on top of each other like for ECS that has Fargate under it, auto scaling, ECS Service, ECS Service Task Definition and ECS Service EIN. You might want to have all of the ECS tiles stacked on top of each other so you can see them still but they aren't the current focus. 
    - [x] Has a hole close to the top that is sized to be combined with the spinable arm on the case for a travel case, easier carry, can be used as a fidget spiner to hold it and can be used to hold it up. 
3. `pocket`: 
    - [ ] Has magnet inserts for usage with a magnetic whiteboard. 
    - [ ] Has a pocket or slot for tiles to fit into. 
4. `case`: 
    - [ ] Carry case for the remaining `tile` & `pocket` components. 
    - [ ] Has magnet inserts for the kit to be stored on a magnetic whiteboard. 
    - [ ] A spinable arm to draw circles
        - Example of a spinning gear: https://www.printables.com/model/909541-fidget-gear-ring
        - Example of a collapsing Katana: https://www.youtube.com/shorts/zxEmTZwg8Pc
        - [ ] Smallest collapsing layer of the arm should have storage for a Dry Erase Marker
        - [ ] End of the arm should hold a dry erase arm with a snapping mechanism or naturally cup a marker for the user to use while tracking.
        - [ ] Bonus: Should have holds that can be used to hold a dry erase marker while rotating at certain lengths maybe every 2 inches or whatever will fit.
    - [ ] A coding indent spacing guide like a multitool ruler.
        - Example of a multitool ruler with 1 spinning rotation: https://www.youtube.com/shorts/IwkuNI8ix4o
            - Would need up to 10-20 layers but is likely limited due to material strength before getting to 10-20 layers.

---

## Previous Prototype w Spinner
Previously was able to get the prototype with larger magnets to work with the following parameters:
- `MAG_D = 15.5;` THIS LIKELY NEEDS TO BE REDUCED BY 0.5 then added to MAG_CLEAR. 
- `MAG_H = 3.0;`
- `MAG_CLEAR = 0.5;` THIS LIKELY NEEDS TO BE INCREASED BY 0.5 and reduced from MAG_D. 


---
