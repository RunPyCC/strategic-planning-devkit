include <lib.scad>;

// ---------------- Parameters ----------------
// ICON_SVG = "icons/misc/nginx.svg"; //override default
ICON_THICKNESS   = ICON_H;       // overall tile thickness
ICON_CLEARANCE   = FIT;          // shrink logo to ensure it fits pocket
USE_SPINNER_HOLE = true;         // include the fidget / carry hole
SPINNER_DIAMETER = SPINNER_D;    // hole diameter

// ---------------- Assembly ----------------
icon_with_features(icon_height   = ICON_THICKNESS,
                   clearance     = ICON_CLEARANCE,
                   use_spinner   = USE_SPINNER_HOLE,
                   spinner_d     = SPINNER_DIAMETER);
