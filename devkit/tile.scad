include <lib.scad>;

// ---------------- Parameters ----------------
// ICON_SVG = "icons/misc/nginx.svg"; //override default
TILE_HEIGHT      = TILE_H;
ICON_DEPTH       = ICON_H;
TILE_CLEARANCE   = FIT;
USE_SPINNER_HOLE = true;
SPINNER_DIAMETER = SPINNER_D;

// NEW: make tile slightly smaller than the pocket
// 1.2mm total reduction = 0.6mm per side
TILE_D = COASTER_D - 2.4; // TEST 2 '2.4' | Next try 2.2mm

// ---------------- Assembly ----------------
tile_with_features(tile_height = TILE_HEIGHT,
                   icon_depth  = ICON_DEPTH,
                   clearance   = TILE_CLEARANCE,
                   use_spinner = USE_SPINNER_HOLE,
                   spinner_d   = SPINNER_DIAMETER,
                   d           = TILE_D);

// Show the two outlines as thin plates so you can SEE the difference
//%linear_extrude(height=0.2) tileoutline2d(COASTER_D);  // reference (full size)
//#linear_extrude(height=0.4) tileoutline2d(TILE_D);     // should be smaller


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

