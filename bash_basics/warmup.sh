#!/bin/bash
# warmup.sh - Daily Bash practice routine

echo "👋 Daily Bash Warmup Starting..."

# 1. Where am I?
pwd

# 2. List files
ls

# 3. List files with details
ls -l

# 4. Go up one folder and back again
cd ..
echo "Now in: $(pwd)"
cd -

# 5. Make a folder
mkdir -p practice_folder
echo "Created practice_folder"

# 6. Move into folder
cd practice_folder

# 7. Create file with text
echo "hello bash world" > hello.txt

# 8. Show file contents
cat hello.txt

# 9. Rename file
mv hello.txt hi.txt
echo "Renamed file to hi.txt"

# 10. Delete file
rm hi.txt
echo "Deleted hi.txt"

# Go back up and remove practice folder
cd ..
rm -r practice_folder

echo "✅ Daily Bash Warmup Finished!"
