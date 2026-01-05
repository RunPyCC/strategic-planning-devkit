include <lib.scad>;

// ---------------- Parameters ----------------
// --- ADDITIONAL ICON_SVG selectors included below
// ICON_SVG = "icons/misc/docker.svg"; //override default
POCKET_HEIGHT      = POCKET_H;     // total height of the pocket coaster
ICON_DEPTH         = ICON_H;       // depth of the recess for the icon
POCKET_CLEARANCE   = FIT;          // radial clearance so the tile fits
USE_SPINNER_HOLE   = true;         // default pocket omits the spinner hole
SPINNER_DIAMETER   = SPINNER_D;    // diameter if spinner hole enabled
USE_MAGNET_CUTOUTS = true;         // include magnet cavities
MAGNET_Z0          = BOTTOM_SKIN;  // distance from bottom to start cavities

// --- NEW: Tile-retention “raised edges” (octagon only) ---
USE_TILE_RETENTION = false;

// How snug the tile sits against the retainer (XY gap).
RET_XY_CLEAR       = 0.25;          // mm (reduce for tighter hold)

// How thick the raised edge is (in XY).
RET_WALL_THICK     = 2.0;           // mm

// ADDED AT A LATER POINT
RET_ANCHOR_Z = 0.6; // mm: how far the lip sinks into the pocket top

// How far the raised edge extends outward from the pocket face (in Z).
// Think: should be >= tile thickness so it actually “catches” it.
RET_Z_DEPTH        = TILE_H + 2; // mm how far the shelf extends outward.

// Small taper so the retainer leans slightly inward as it extends outward.
// Keep small: 0.005–0.020 is typical.
RET_DRAFT          = 0.010;         // unitless scale fraction

// Bottom sector = bottom 3 edges for a SHAPE_ROT=22.5 octagon.
// (Rays at 225° and 315° around the origin.)
RET_ANGLE0         = 225;
RET_ANGLE1         = 315;


// ---------------- Helpers ----------------
module _sector2d(a0=225, a1=315, d=COASTER_D, r_factor=3) {
    r = d * r_factor;
    polygon(points=[
        [0,0],
        [r*cos(a0), r*sin(a0)],
        [r*cos(a1), r*sin(a1)]
    ]);
}

// How much to taper ONLY the inner (tile-facing) wall.
// 0.005–0.030 is a reasonable range.
RET_INNER_DRAFT = 0.015;

// Outer boundary of the lip sector (vertical wall)
module _ret_outer2d(d=COASTER_D, inset=0, a0=RET_ANGLE0, a1=RET_ANGLE1) {
    intersection() {
        offset(delta = -inset) tileoutline2d(d);
        _sector2d(a0=a0, a1=a1, d=d);
    }
}

// Inner boundary that gets subtracted (this defines the tile-facing wall)
module _ret_inner2d(d=COASTER_D, inset=0, thick=RET_WALL_THICK, a0=RET_ANGLE0, a1=RET_ANGLE1) {
    intersection() {
        offset(delta = -(inset + thick)) tileoutline2d(d);
        _sector2d(a0=a0, a1=a1, d=d);
    }
}

// Lip where ONLY the inner face is tapered
module tile_retainer(pocket_height=POCKET_HEIGHT,
                     d=COASTER_D,
                     z_depth=RET_Z_DEPTH,
                     a0=RET_ANGLE0,
                     a1=RET_ANGLE1,
                     inset=0,
                     anchor=RET_ANCHOR_Z,
                     inner_draft=RET_INNER_DRAFT) {

    translate([0,0,pocket_height - anchor])
        difference() {
            // Outer wall: straight/vertical
            linear_extrude(height=z_depth + anchor)
                _ret_outer2d(d=d, inset=inset, a0=a0, a1=a1);

            // Inner void: slightly larger at the top -> creates slant ONLY on tile side
            linear_extrude(height=z_depth + anchor, scale=1 - inner_draft)
                _ret_inner2d(d=d, inset=inset, thick=RET_WALL_THICK, a0=a0, a1=a1);
        }
}

// ---------------- Assembly ----------------
union() {
    pocket_with_features(pocket_height = POCKET_HEIGHT,
                         icon_depth    = ICON_DEPTH,
                         clearance     = POCKET_CLEARANCE,
                         use_spinner   = USE_SPINNER_HOLE,
                         spinner_d     = SPINNER_DIAMETER,
                         use_magnets   = USE_MAGNET_CUTOUTS,
                         magnet_z0     = MAGNET_Z0);

    // Only apply this to the octagon pocket
    if (USE_TILE_RETENTION && (SHAPE == "octagon"))
        tile_retainer();
}


// ---------------- ICON_SVG Selector -------
// ------------------ icons/api/ ------------
// ICON_SVG = "icons/api/fastapi.svg"; //override default
// ICON_SVG = "icons/api/flask.svg"; 
// ------------------ SKIPPED icons/aws/ ----
// ------------------ icons/db/ -------------
// ICON_SVG = "icons/db/alembic.svg"; 
// ICON_SVG = "icons/db/dbeaver.svg"; 
// ICON_SVG = "icons/db/sqlalchemy.svg"; 
// ICON_SVG = "icons/db/sqlite.svg"; 
// ------------------ icons/frontend/ -------
// ICON_SVG = "icons/frontend/nextjs.svg"; 
// ICON_SVG = "icons/frontend/svelte.svg"; 
// ------------------ icons/ide/ ------------
// ICON_SVG = "icons/ide/gnu-emacs.svg"; 
// ICON_SVG = "icons/ide/vs-code.svg"; 
// ------------------ icons/misc/ -----------
// ICON_SVG = "icons/misc/digital-ocean.svg"; 
// ICON_SVG = "icons/misc/django.svg"; 
// ICON_SVG = "icons/misc/docker.svg"; 
// ICON_SVG = "icons/misc/gunicorn.svg"; 
// ICON_SVG = "icons/misc/nginx.svg"; 
// ICON_SVG = "icons/misc/npm.svg"; 
// ICON_SVG = "icons/misc/postgresql.svg"; 
// ICON_SVG = "icons/misc/proxmox.svg"; 
// ICON_SVG = "icons/misc/users.svg"; 
// ICON_SVG = "icons/misc/wordpress.svg"; 
// ------------------ icons/aws/ -----------
// ICON_SVG = "icons/aws/aws-analytics.svg"; 
// ICON_SVG = "icons/aws/aws-athena.svg"; 
// ICON_SVG = "icons/aws/aws-auto-scaling.svg"; 
// ICON_SVG = "icons/aws/aws-batch.svg"; 
// ICON_SVG = "icons/aws/aws-cli.svg"; 
// ICON_SVG = "icons/aws/aws-cloudfront.svg"; 
// ICON_SVG = "icons/aws/aws-cloudhsm.svg"; 
// ICON_SVG = "icons/aws/aws-cloudsearch-search-documents.svg"; 
// ICON_SVG = "icons/aws/aws-cloudsearch.svg"; 
// ICON_SVG = "icons/aws/aws-cloudtrail.svg"; 
// ICON_SVG = "icons/aws/aws-cloudwatch.svg"; 
// ICON_SVG = "icons/aws/aws-cognito.svg"; 
// ICON_SVG = "icons/aws/aws-data-pipeline.svg"; 
// ICON_SVG = "icons/aws/aws-directory-service.svg"; 
// ICON_SVG = "icons/aws/aws-dms-database-migration-workflow.svg"; 
// ICON_SVG = "icons/aws/aws-dms.svg"; 
// ICON_SVG = "icons/aws/aws-dynamodb-attribute.svg"; 
// ICON_SVG = "icons/aws/aws-dynamodb-attributes.svg"; 
// ICON_SVG = "icons/aws/aws-dynamodb-global-secondary-index.svg"; 
// ICON_SVG = "icons/aws/aws-dynamodb-item.svg"; 
// ICON_SVG = "icons/aws/aws-dynamodb-table.svg"; 
// ICON_SVG = "icons/aws/aws-dynamodb.svg"; 
// ICON_SVG = "icons/aws/aws-ebs-snapshot.svg"; 
// ICON_SVG = "icons/aws/aws-ebs-volume.svg"; 
// ICON_SVG = "icons/aws/aws-ebs.svg"; 
// ICON_SVG = "icons/aws/aws-ec2-ami.svg"; 
// ICON_SVG = "icons/aws/aws-ec2-elastic-ip-addr.svg"; 
// ICON_SVG = "icons/aws/aws-ec2.svg"; 
// ICON_SVG = "icons/aws/aws-ecr.svg"; 
// ICON_SVG = "icons/aws/aws-ecs-kubernetes.svg"; 
// ICON_SVG = "icons/aws/aws-ecs-svc-task-ct.svg"; 
// ICON_SVG = "icons/aws/aws-ecs-svc-task.svg"; 
// ICON_SVG = "icons/aws/aws-ecs-svc.svg"; 
// ICON_SVG = "icons/aws/aws-ecs.svg"; 
// ICON_SVG = "icons/aws/aws-elastic-beanstalk-app.svg"; 
// ICON_SVG = "icons/aws/aws-elastic-beanstalk-deploy.svg"; 
// ICON_SVG = "icons/aws/aws-elasticsearch-service.svg"; 
// ICON_SVG = "icons/aws/aws-emr-cluster.svg"; 
// ICON_SVG = "icons/aws/aws-emr-engine-mapr-m3.svg"; 
// ICON_SVG = "icons/aws/aws-emr-engine-mapr-m5.svg"; 
// ICON_SVG = "icons/aws/aws-emr-engine-mapr-m7.svg"; 
// ICON_SVG = "icons/aws/aws-emr-engine.svg"; 
// ICON_SVG = "icons/aws/aws-emr-hdfs-cluster.svg"; 
// ICON_SVG = "icons/aws/aws-emr.svg"; 
// ICON_SVG = "icons/aws/aws-gen-compute.svg"; 
// ICON_SVG = "icons/aws/aws-glue-crawlers.svg"; 
// ICON_SVG = "icons/aws/aws-glue-data-catalog.svg"; 
// ICON_SVG = "icons/aws/aws-glue.svg"; 
// ICON_SVG = "icons/aws/aws-kinesis-data-analytics.svg"; 
// ICON_SVG = "icons/aws/aws-kinesis-data-firehose.svg"; 
// ICON_SVG = "icons/aws/aws-kinesis-data-streams.svg"; 
// ICON_SVG = "icons/aws/aws-kinesis-video-streams.svg"; 
// ICON_SVG = "icons/aws/aws-kinesis.svg"; 
// ICON_SVG = "icons/aws/aws-lake-formation.svg"; 
// ICON_SVG = "icons/aws/aws-lambda.svg"; 
// ICON_SVG = "icons/aws/aws-lightsail.svg"; 
// ICON_SVG = "icons/aws/aws-managed-streaming-for-kafka.svg"; 
// ICON_SVG = "icons/aws/aws-quicksight.svg"; 
// ICON_SVG = "icons/aws/aws-rds.svg"; 
// ICON_SVG = "icons/aws/aws-redshift-dense-compute-node.svg"; 
// ICON_SVG = "icons/aws/aws-redshift-dense-storage-node.svg"; 
// ICON_SVG = "icons/aws/aws-redshift.svg"; 
// ICON_SVG = "icons/aws/aws-users.svg"; 
// ICON_SVG = "icons/aws/aws-vmware-cloud.svg"; 
// ICON_SVG = "icons/aws/aws-vpc-igw.svg"; 
// ICON_SVG = "icons/aws/aws-vpc.svg"; 
// ICON_SVG = "icons/aws/aws-vpce.svg"; 


// ---------------- Display -----------------
//!magnet_cavities(points=[[-1,0],[1,0],[0,-1]]);
