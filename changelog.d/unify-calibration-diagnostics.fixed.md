Use one typed schema and atomic writer for calibration diagnostics, make new
US local and national diagnostics schema 8, and record diagnostics failures
without aborting dataset publication. UK calibration producers now validate and
write schema-8 diagnostics through the same implementation, while the dense UK
release assembler validates and copies those canonical bytes without rewriting
the document.
