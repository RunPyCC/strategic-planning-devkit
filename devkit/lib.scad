// ------------------------------------------------
//  Shared geometry helpers for model_tiles components
// ------------------------------------------------

// ------------------------------------------------
// How to think about MAG_D vs MAG_H.
// From the perspective of how the shape would be printed. For example, MAG_H is the magnet thinkness which makes sense when you think about how it will be placed on the 3D printer when it is time to print. MAG_D makes sense when thought of from a bird's eye view. 
// ------------------------------------------------

// --- Shape and sizing parameters ----------------
SHAPE           = "octagon";    // "circle" or "octagon"
SHAPE_ROT       = 22.5;         // rotate outline (deg)
COASTER_D       = 101.6;        // 4 in → mm (circle dia or octagon across-flats)

// --- Thickness / depth parameters ---------------
CASE_H          = 5.0;          // default case thickness. | #TODO - [ ] refac for base carrying case
ICON_H          = 0.5;          // default icon thickness.
TILE_H          = 1.2;          // default tile thickness.
POCKET_H        = 3.0;          // default pocket thickness. | Testing 3.0 = buffer 0.2 + MAG_H 2.0 + ICON_H 0.5 + BOTTOM_SKIN 0.3  | prev 5.0

// --- Icon source (override by defining ICON_SVG before include <lib.scad>) ---
ICON_SVG = "icons/misc/docker.svg";

// --- Icon sizing --------------------------------
MARGIN          = 30.0;         // edge margin for icon
CLEARANCE       = 0.10;         // total gap between pocket & inlay (mm)
FIT             = CLEARANCE/2;  // per-side clearance

// --- Magnet parameters --------------------------
MAG_D           = 6.0;          // disc magnet diameter - prev 15.5 but likely should have been 15.0
MAG_H           = 2.0;          // magnet thickness - prev 3.0
MAG_CLEAR       = 0.6;          // extra diameter clearance for easy fit - prev 0.5 but likely should have been 1.0
BOTTOM_SKIN     = 0.3;          // plastic between magnet and the underside
MAG_EDGE_CLEAR  = 4.0;          // sole purpose for gap btw magnets to outer edge
EDGE_CLEAR      = 15;           // Possibly used multiple places... before was: desired gap from magnets to outer edge

// --- Magnet Grid Region parameters --------------
MAG_REGION_D    = 15.2;         // diameter for the magnet grid region
MAG_GRID_GAP    = 2.0;          // gap btw magnets in the magnet grid
REGION_PAD      = 0.0;          // extra slack around cluster (mm). Try 0, 0.5, 1.0
cavity_d        = MAG_D + MAG_CLEAR;

// Tight rectangle that fits a 2x2 cluster with MAG_GRID_GAP between cavities
MAG_REGION_SIZE = [
  2*cavity_d + MAG_GRID_GAP + 2*REGION_PAD,  // width
  2*cavity_d + MAG_GRID_GAP + 2*REGION_PAD   // height
];

// Size helper for arbitrary magnet grids (cols span X, rows span Y)
function magnet_grid_region_size(rows,
                                 cols,
                                 diameter=MAG_D,
                                 clearance=MAG_CLEAR,
                                 gap=MAG_GRID_GAP,
                                 pad=REGION_PAD) =
    let (cavity_d = diameter + clearance)
        [cols*cavity_d + (cols - 1)*gap + 2*pad,
         rows*cavity_d + (rows - 1)*gap + 2*pad];


// --- Spinner parameters -------------------------
SPINNER_D       = 20;           // diameter of the finger-through hole (mm)

// --- Globals ------------------------------------
SEG             = 200;          // smoothness for circles
$fn             = SEG;

// ------------------------------------------------
// 2D geometry helpers
module icon2d_raw() { import(ICON_SVG, center=true); }
module icon2d_sized()   {
    resize([COASTER_D - 2*MARGIN, COASTER_D - 2*MARGIN, 0], auto=true)
        icon2d_raw();
}
module pocket2d(clearance=FIT) { offset(delta=+clearance) icon2d_sized(); }
module inlay2d(clearance=FIT)  { offset(delta=-clearance) icon2d_sized(); }
module ngon2d(n, across_flats) {
    r = (across_flats/2)/cos(180/n); // OpenSCAD trig uses degrees
    rotate(SHAPE_ROT)
        polygon(points=[for (i=[0:n-1]) [ r*cos(360*i/n), r*sin(360*i/n) ]]);
}
module tileoutline2d() {
    if (SHAPE == "octagon")  ngon2d(8, COASTER_D);
    else                     circle(d=COASTER_D);
}

// ------------------------------------------------
// 3D building blocks
module tile_base(height=TILE_H) {
    linear_extrude(height=height) tileoutline2d();
}

module pocket_base(height=POCKET_H) {
    linear_extrude(height=height) tileoutline2d();
}

module case_base(height=CASE_H) {
    linear_extrude(height=height) tileoutline2d();
}

module icon_solid(height=ICON_H, clearance=FIT) {
    linear_extrude(height=height) inlay2d(clearance);
}

module tile_solid(height=TILE_H, clearance=FIT) {
    linear_extrude(height=height) inlay2d(clearance);
}

module spinner_cut(height, diameter=SPINNER_D) {
    // Extend slightly above & below to guarantee a clean boolean
    translate([0, 32.3, -0.1])
        cylinder(d = diameter, h = height + 0.2, center = false);
}

module tile_cut(depth=TILE_H, tile_height=TILE_H, clearance=FIT) {
    translate([0, 0, tile_height - depth])
        linear_extrude(height=depth + 0.05) pocket2d(clearance);
}

// ------------------------------------------------
// Magnet building blocks
module magnet_cavity(diameter=MAG_D, clearance=MAG_CLEAR, height=MAG_H) {
    cylinder(d = diameter + clearance,
             h = height + 0.2,
             center = false);
}

module magnet_cavity_grid(rows=2,
                          cols=2,
                          spacing_x=undef, //when left as undef, the grid spans the tile/pocket
                          spacing_y=undef, //when left as undef, the grid spans the tile/pocket
                          z0=BOTTOM_SKIN,
                          diameter=MAG_D,
                          clearance=MAG_CLEAR,
                          height=MAG_H,
                          origin=[0, 0],
                          rows_enabled=undef,
                          cols_enabled=undef) {
    assert(rows >= 1 && cols >= 1, "rows and cols must be at least 1");

    available = (COASTER_D - 2*MAG_EDGE_CLEAR) - diameter;
    pitch_x = is_undef(spacing_x)
        ? (cols > 1 ? available/(cols - 1) : 0)
        : spacing_x;
    pitch_y = is_undef(spacing_y)
        ? (rows > 1 ? available/(rows - 1) : 0)
        : spacing_y;

    span_x = pitch_x * (cols - 1);
    span_y = pitch_y * (rows - 1);
    max_offset = (COASTER_D/2) - MAG_EDGE_CLEAR - (diameter/2);
    assert(span_x/2 <= (max_offset + 1e-3),
           "Magnet grid exceeds available width; adjust MAG_EDGE_CLEAR, spacing_x, or cols.");
    assert(span_y/2 <= (max_offset + 1e-3),
           "Magnet grid exceeds available height; adjust MAG_EDGE_CLEAR, spacing_y, or rows.");

    row_indices = is_undef(rows_enabled) ? [0:rows-1] : rows_enabled;
    col_indices = is_undef(cols_enabled) ? [0:cols-1] : cols_enabled;

    for (r=row_indices)
        for (c=col_indices)
            translate([ origin[0] - span_x/2 + c*pitch_x,
                        origin[1] - span_y/2 + r*pitch_y,
                        z0 ])
                magnet_cavity(diameter, clearance, height);
}

// DEV WORKSPACE
// Places children() at each legacy magnet "region" center.
// sx_values / sy_values let you enable subsets (like your spinner logic).
module magnet_regions(region_size=MAG_REGION_SIZE,
                      points=undef,
                      sx_values=[-1, 1],
                      sy_values=[-1, 1]) {

    // How far the REGION CENTER can be pushed toward each edge
    off_x = (COASTER_D/2) - MAG_EDGE_CLEAR - (region_size[0]/2);
    off_y = (COASTER_D/2) - MAG_EDGE_CLEAR - (region_size[1]/2);

    if (!is_undef(points)) {
        for (p = points)
            translate([p[0]*off_x, p[1]*off_y, 0])
                children();
    } else {
        for (sy=sy_values)
            for (sx=sx_values)
                translate([sx*off_x, sy*off_y, 0])
                    children();
    }
}

// Generic local cluster helper for rectangular grids.
module magnet_cluster_local(rows,
                            cols,
                            z0=BOTTOM_SKIN,
                            diameter=MAG_D,
                            clearance=MAG_CLEAR,
                            height=MAG_H,
                            gap=MAG_GRID_GAP,
                            region_size=magnet_grid_region_size(rows, cols, diameter, clearance, gap, REGION_PAD)) {

    cavity_d = diameter + clearance;
    pitch_desired = cavity_d + gap;

    span_limit_x = region_size[0] - cavity_d;
    span_limit_y = region_size[1] - cavity_d;

    pitch_x = cols > 1 ? min(pitch_desired, span_limit_x/(cols - 1)) : 0;
    pitch_y = rows > 1 ? min(pitch_desired, span_limit_y/(rows - 1)) : 0;

    assert(pitch_x >= 0 && pitch_y >= 0,
        "Region too small for requested magnet cluster; increase region padding or shrink magnet dimensions.");

    magnet_cavity_grid(rows=rows, cols=cols,
                       spacing_x=pitch_x,
                       spacing_y=pitch_y,
                       origin=[0,0],
                       z0=z0,
                       diameter=diameter,
                       clearance=clearance,
                       height=height);
}

// A compact 2x2 cluster that fits inside a circular "region" (legacy magnet footprint).
// Assumes it is already positioned by translate(), so origin is local [0,0].
module magnet_cluster_2x2_local(z0=BOTTOM_SKIN,
                                diameter=MAG_D,
                                clearance=MAG_CLEAR,
                                height=MAG_H,
                                gap=MAG_GRID_GAP,
                                region_size=MAG_REGION_SIZE) {

    magnet_cluster_local(rows=2, cols=2,
                         z0=z0,
                         diameter=diameter,
                         clearance=clearance,
                         height=height,
                         gap=gap,
                         region_size=region_size);
}

// A 3x2 (or 2x3) cluster that runs from the outer edge toward the center.
// Swap rows/cols when you want the longer run to point vertically.
module magnet_cluster_3x2_local(z0=BOTTOM_SKIN,
                                rows=2,
                                cols=3,
                                diameter=MAG_D,
                                clearance=MAG_CLEAR,
                                height=MAG_H,
                                gap=MAG_GRID_GAP,
                                region_size=magnet_grid_region_size(rows, cols, diameter, clearance, gap, REGION_PAD)) {

    magnet_cluster_local(rows=rows, cols=cols,
                         z0=z0,
                         diameter=diameter,
                         clearance=clearance,
                         height=height,
                         gap=gap,
                         region_size=region_size);
}


// This is the new meaning of "magnet_cavities":
// put a 2x2 cluster *where each legacy magnet used to be*.
module magnet_cavities(z0=BOTTOM_SKIN,
                       diameter=MAG_D,
                       clearance=MAG_CLEAR,
                       height=MAG_H,
                       region_size=MAG_REGION_SIZE,
                       layout="2x2",  // (2) opts: "2x2" or "edge_3x2"
                       points=undef,
                       sx_values=[-1, 1],
                       sy_values=[-1, 1]) {
    resolved_points = is_undef(points)
        ? [for (sy=sy_values) for (sx=sx_values) [sx, sy]]
        : points;

    if (layout == "edge_3x2") {
        for (p = resolved_points) {
            is_vertical_side = abs(p[0]) >= abs(p[1]);
            local_rows = is_vertical_side ? 2 : 3;
            local_cols = is_vertical_side ? 3 : 2;
            local_region = magnet_grid_region_size(rows=local_rows,
                                                  cols=local_cols,
                                                  diameter=diameter,
                                                  clearance=clearance,
                                                  gap=MAG_GRID_GAP,
                                                  pad=REGION_PAD);

            magnet_regions(region_size=local_region, points=[p])
                magnet_cluster_3x2_local(z0=z0,
                                        rows=local_rows,
                                        cols=local_cols,
                                        diameter=diameter,
                                        clearance=clearance,
                                        height=height,
                                        gap=MAG_GRID_GAP,
                                        region_size=local_region);
        }
    } else {
        magnet_regions(region_size=region_size,
                       points=resolved_points)
            magnet_cluster_2x2_local(z0=z0,
                                     diameter=diameter,
                                     clearance=clearance,
                                     height=height,
                                     gap=MAG_GRID_GAP,
                                     region_size=region_size);
    }
}


// ------------------------------------------------
// Component assemblies
module icon_with_features(icon_height=ICON_H,
                          clearance=FIT,
                          use_spinner=true,
                          spinner_d=SPINNER_D) {
    difference() {
        icon_solid(icon_height, clearance);
        if (use_spinner) spinner_cut(icon_height, spinner_d);
    }
}

module tile_with_features(tile_height=TILE_H,
                            icon_depth=ICON_H,
                            clearance=FIT,
                            use_spinner=false,
                            spinner_d=SPINNER_D) {
    assert(tile_height > icon_depth,
           "Icon depth must be less than the tile height.");
    difference() {
        tile_base(tile_height);
        tile_cut(icon_depth, tile_height, clearance);
        if (use_spinner) spinner_cut(tile_height, spinner_d);
    }
}

module pocket_with_features(pocket_height=POCKET_H,
                            icon_depth=ICON_H,
                            clearance=FIT,
                            use_spinner=false,
                            spinner_d=SPINNER_D,
                            use_magnets=true,
                            magnet_z0=BOTTOM_SKIN,
                            magnet_layout="2x2", // (2) opts: "2x2" or "edge_3x2"
                            magnet_diameter=MAG_D,
                            magnet_clearance=MAG_CLEAR,
                            magnet_height=MAG_H) {
    assert(pocket_height > icon_depth,
           "Icon depth must be less than the pocket height.");
    if (use_magnets)
        assert(pocket_height > (magnet_z0 + magnet_height + 0.1),
               "Increase pocket height or adjust magnet parameters.");
    difference() {
        pocket_base(pocket_height);
        tile_cut(icon_depth, pocket_height, clearance);
        if (use_spinner) spinner_cut(pocket_height, spinner_d);
        if (use_magnets)
            magnet_cavities(z0=magnet_z0,
                            layout=magnet_layout,
                            diameter=magnet_diameter,
                            clearance=magnet_clearance,
                            height=magnet_height,
                            points=[[-1,0],[1,0],[0,-1]]);
    }
}

module case_with_features(case_height=CASE_H,
                          use_spinner=true,
                          spinner_d=SPINNER_D,
                          use_magnets=true,
                          magnet_z0=BOTTOM_SKIN,
                          magnet_layout="2x2", // (2) opts: "2x2" or "edge_3x2"
                          magnet_diameter=MAG_D,
                          magnet_clearance=MAG_CLEAR,
                          magnet_height=MAG_H) {
    if (use_magnets)
        assert(case_height > (magnet_z0 + magnet_height + 0.1),
               "Increase base height or adjust magnet parameters.");
    difference() {
        case_base(case_height);
        if (use_spinner) spinner_cut(case_height, spinner_d);
        if (use_magnets)
            magnet_cavities(z0=magnet_z0,
                            layout=magnet_layout,
                            diameter=magnet_diameter,
                            clearance=magnet_clearance,
                            height=magnet_height,
                            sy_values = use_spinner ? [-1] : [-1, 1]);
    }
}
