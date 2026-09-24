Use one typed schema and atomic writer for calibration diagnostics, make new
US local and national diagnostics schema 8, and record diagnostics failures
without aborting dataset publication. UK calibration producers now validate and
write schema-8 diagnostics through the same implementation, while the dense UK
release assembler validates and copies those canonical bytes without rewriting
the document. The canonical schema now defines every stable field in the UK
extension and validates its count, ratio, pass-rate, area-summary, and rotated-
holdout relationships at construction; only historical schema-6 and schema-7
UK artifacts use the isolated compatibility validator. Current models reject
non-finite numbers recursively, and the canonical writer uses strict JSON
serialization so `NaN` and infinities cannot silently become `null`.
