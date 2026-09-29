Separate country-owned staging repository defaults from shared command-line,
remote-storage, and repeated-upload failure handling while preserving the US
version 1 file contract. Use one country-neutral stage-observation lifecycle
for stage plans and graph transforms. Require runs to finish before authenticated
readback, reject content changes after completion or failure, record every
national-calibration operation with elapsed time, and propagate repository
rate-limit and server errors during access verification. Reject scalar-valued
individual records from reviewed aggregate artifacts.
