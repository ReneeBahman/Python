#!/bin/bash
# experiments.sh - Sandbox for trying Bash commands safely

echo "🧪 Welcome to Bash Experiments"
echo "This is your sandbox - nothing important here, play as you like."

# --- Basic info ---
echo "Current folder:"
pwd

echo "Files here:"
ls

# --- Play zone: create a playground folder ---
mkdir -p play_zone
cd play_zone
echo "Now inside: $(pwd)"

# --- Sample experiments ---
echo "Creating some files..."
echo "file one" > file1.txt
echo "file two" > file2.txt

echo "Listing files:"
ls -l

# --- Try a loop ---
echo "Looping through files:"
for f in *.txt; do
  echo "File: $f, Contents: $(cat $f)"
done

# --- Cleanup (optional, comment this out if you want to keep files) ---
cd ..
rm -r play_zone
echo "Cleaned up play_zone"

echo "✅ Experiments finished! Add more commands here anytime."
