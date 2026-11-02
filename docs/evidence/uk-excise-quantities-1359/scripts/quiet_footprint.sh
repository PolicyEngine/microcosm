# Sourced by the 1359 GSGB scripts. big_footprints prints every process whose memory
# footprint (top's MEM: resident plus compressed and swapped) is above $1 MB, other than
# the PIDs in $2 (space-separated). ps rss misses a swapped-out process, so it is not used.
big_footprints() {
  local floor=$1 skip=" ${2:-} "
  top -l 1 -o mem -n 30 -stats pid,mem,command 2>/dev/null | awk -v floor="$floor" -v skip="$skip" '
    $1 ~ /^[0-9]+$/ {
      m = $2; gsub(/[+-]$/, "", m)
      unit = substr(m, length(m), 1); value = substr(m, 1, length(m) - 1) + 0
      if (unit == "G") value *= 1024; else if (unit == "K") value /= 1024; else if (unit == "B") value /= 1048576; else if (unit != "M") value = m / 1048576
      if (value > floor && index(skip, " " $1 " ") == 0) print $1, $2, $3
    }'
}
