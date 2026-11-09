#!/bin/bash
#STEP 0 : load rx_tools "module load rxtools"
#FOR SOME REASON THIS SCRIPT ONLY RUNS WELL within the SCRATCH FOLDER
#COPY IN THE SCRATCH AND THEN SEND A SLURM , DO NOT RUN ON THE LOGIN NODE

#STEP 1: BRING THE DATA FROM MFS

DATE=$(date +"%Y%m%d")
#to force specific day:
DATE="20241010"
sc4003_DIR="/mfs/io/groups/uars/scintpi/sc002"
ORIGIN_DIR="$sc4003_DIR/$DATE/"
SOURCE_DIR="/home/jxg200016/scratch"
#copying data from origin dir to source dir
rsync "$ORIGIN_DIR"cr01*.2*_ "$SOURCE_DIR"/
echo "origin: $ORIGIN_DIR"






#STEP 2: APPLY BIN 2 ASC
# Define source and target directories
#SOURCE_DIR="/home/jxg200016/scratch"
TARGET_DIR="/home/jxg200016/scratch"

# Command to run if the file is missing in target (you can customize this)
# For example, we'll just echo the missing file path, or you could copy it, log it, etc.
#MISSING_CMD="/opt/ohpc/pub/apps/rxtools/bin/bin1asc -f"
MISSING_CMD="rxtools_exec.sh bin2asc -f "

# Loop over each file in the source directory
for SRC_FILE in "$SOURCE_DIR"/*.2*_; do
    #skip if no .25_ files found
    [ -f "$SRC_FILE" ] || continue
    # Get just the filename (basename)
    FILENAME=$(basename "$SRC_FILE")
    BASE=$(echo "$FILENAME" | sed -E 's/\.[0-9]+_$//')
    echo "base: $BASE "
    echo "file: $FILENAME"
    #echo "Extension: $EXT"
    TXT_FILENAME=$(echo "$TARGET_DIR/$BASE.txt") # Build the path of the expected file in the target directory
    echo "txtname: $TXT_FILENAME"
    # Check if the file exists in the target directory
    if [ -f "$TXT_FILENAME" ]; then
	echo "Found matching txt for $BASE"
    else
	echo "no TXT found for $BASE"
        # File is missing — run the custom command
        $MISSING_CMD "$SOURCE_DIR/$FILENAME" "--extractGenMeas"
	#-m -o "$TARGET_DIR/$BASE.txt"
    fi
done

