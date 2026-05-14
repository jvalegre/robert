#!/bin/bash

# Get the application directory
APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="$APP_DIR/postinstall.log"

echo "[+] Starting macOS postinstall..." > "$LOG_FILE"

# Create environment directory
mkdir -p "$APP_DIR/Contents/Resources/robert_env" >> "$LOG_FILE" 2>&1

# Extract conda environment
echo "[+] Extracting conda environment..." >> "$LOG_FILE"
tar -xzf "$APP_DIR/Contents/Resources/robert_env.tar.gz" -C "$APP_DIR/Contents/Resources/robert_env" >> "$LOG_FILE" 2>&1

# Activate and unpack environment
echo "[+] Unpacking conda environment..." >> "$LOG_FILE"
source "$APP_DIR/Contents/Resources/robert_env/bin/activate" >> "$LOG_FILE" 2>&1
conda-unpack >> "$LOG_FILE" 2>&1

# Cleanup
rm "$APP_DIR/Contents/Resources/robert_env.tar.gz" >> "$LOG_FILE" 2>&1

echo "[+] Post-installation complete." >> "$LOG_FILE" 2>&1
