#!/bin/zsh
# The corrected-rent states whose baseline step passed the 8 GiB stop (New York, Florida, Texas, California), each in household
# parts through acs_parts.sh, one process at a time. The corrected-rent baseline computes SNAP and housing subsidy too, so it
# needs more parts than the published-rent pass did.
SP=${0:a:h}; cd $SP
LOG=run/acs_parts.log
for f k in 36 3 12 3 48 4 6 5; do
  zsh acs_parts.sh $f acsrent $k || print -r -- "$(date -u +%TZ) $f acsrent parts failed" >> $LOG
done
print -r -- "$(date -u +%TZ) corrected-rent parts ended" >> $LOG
