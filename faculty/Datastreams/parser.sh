#!/bin/bash

cat Raw?.txt | grep 'hreflang="en">' | sed '/<\/a>$/!d' | sed -r 's#.*>(.*)<\/a>#\1#' > Names1.txt
#Step 1 - reads all files Raw?.txt, where ? is a wildcard character
#Step 2 - culls all lines which don't include the string hreflang="en">
#Step 3 - culls all lines which don't have the closing HTML tag </a>
#Step 4 - replaces all strings which fit this pattern: .*>(.*)<\/a>
#	with the first container group, which in this case is the
#	thing which is in between the > and the <\a>
#Step 5 - Create or overwrite to a file called Names1.txt
cat Raw?.txt | grep 'mailto' | sed -e 's#.*mailto:\(.*\)".*#\1#' > Emails1.txt
#Step 1 - reads all files Raw?.txt, where ? is a wildcard character
#Step 2 - culls all lines which don't include the string mailto
#Step 3 - replaces all strings which fit this pattern: .*mailto:\(.*\)".*
#	again with the first container group, just like last time.
#	The opening bound is mailto: and the closing bound is "
#Step 5 - Create or overwrite to a file called Emails1.txt
paste -d ',' Names1.txt Emails1.txt > merged.csv
#Combines the two files Names1.txt and Emails1.txt into a new file called merged.csv, separated by a comma
sed '1i\site title,email found' merged.csv | sed 'G' > ../osu-emails.csv
#Step 1 - Add the title entries site title and email found
#Step 2 - Add an extra line between every line
#Step 3 - Outputs to osu-emails.csv in the directory above
